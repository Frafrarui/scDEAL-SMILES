#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bulkmodel_smiles.py
====================
Multi-drug bulk model training with SMILES integration.

Differences from original bulkmodel.py:
  - Trains with ALL 83 drugs simultaneously (not one at a time)
  - Each sample is (gene_expression, smiles_tokens, label)
  - Dataset expands from ~1280 to ~106K samples (cell × drug pairs)
  - Uses PretrainedPredictorWithDrug instead of PretrainedPredictor
  - Uses train_predictor_model_smiles from trainers.py

Usage:
  # Full training:
  python bulkmodel_smiles.py --epochs 500 --dimreduce DAE --device cpu

  # Dry-run (quick local test, ~30 seconds):
  python bulkmodel_smiles.py --epochs 2 --dimreduce DAE --device cpu --dry_run True
"""

import argparse
import logging
import sys
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import (average_precision_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import trainers as t
import utils as ut
import sampling as sam
from models import (AEBase, PretrainedPredictorWithDrug)
from smiles_encoder import prepare_smiles_data

import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_multidrug_dataset(data_r, label_r, drug_tokens, matched_drugs, 
                             na_value=1, dry_run=False):
    """
    Expand the bulk data into a multi-drug dataset.
    
    For each (cell_line_i, drug_j) pair where label is valid:
      → one sample: (expression_i, smiles_tokens_j, label_ij)
    
    Labels are encoded as: sensitive → 0, resistant → 1
    Entries that are NaN or equal to na_value after fillna are excluded.
    """
    if dry_run:
        matched_drugs = matched_drugs[:3]
        print(f"[DRY-RUN] Using only {len(matched_drugs)} drugs: {matched_drugs}")
    
    # Align indices between expression and label data
    common_idx = data_r.index.intersection(label_r.index)
    data_r = data_r.loc[common_idx]
    label_r = label_r.loc[common_idx]
    
    # Label encoding map (SC convention: sensitive=1, resistant=0)
    label_map = {'sensitive': 1, 'resistant': 0}
    
    gene_rows = []
    drug_rows = []
    label_rows = []
    
    for drug_name in matched_drugs:
        if drug_name not in label_r.columns:
            continue
        
        drug_labels = label_r[drug_name]
        
        # Filter: keep only valid string labels (sensitive/resistant)
        # After fillna(1), NaN entries become integer 1 → exclude those
        valid_mask = drug_labels.isin(label_map.keys())
        valid_indices = valid_mask[valid_mask].index
        
        if len(valid_indices) == 0:
            continue
        
        # Get gene expression and encode labels
        gene_data = data_r.loc[valid_indices].values
        labels = drug_labels.loc[valid_indices].map(label_map).values
        
        # Get SMILES tokens for this drug (repeated for each cell line)
        tokens = drug_tokens[drug_name].numpy()
        drug_tokens_repeated = np.tile(tokens, (len(valid_indices), 1))
        
        gene_rows.append(gene_data)
        drug_rows.append(drug_tokens_repeated)
        label_rows.append(labels)
    
    X_gene = np.vstack(gene_rows).astype(np.float32)
    X_drug = np.vstack(drug_rows).astype(np.int64)
    Y = np.concatenate(label_rows).astype(np.int64)
    
    print(f"[DATASET] Built multi-drug dataset:")
    print(f"  Samples: {X_gene.shape[0]:,}")
    print(f"  Genes:   {X_gene.shape[1]:,}")
    print(f"  Drugs:   {len(matched_drugs)}")
    print(f"  SMILES token length: {X_drug.shape[1]}")
    print(f"  Label distribution: sensitive(0)={np.sum(Y==0):,}, resistant(1)={np.sum(Y==1):,}")
    
    return X_gene, X_drug, Y


def run_main(args):
    t0 = time.time()
    
    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck
    na = args.missing_value
    data_path = args.data
    label_path = args.label
    smiles_path = args.smiles_file
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    log_path = args.log
    batch_size = args.batch_size
    encoder_hdims = list(map(int, args.encoder_h_dims.split(",")))
    preditor_hdims = list(map(int, args.predictor_h_dims.split(",")))
    reduce_model = args.dimreduce
    dry_run = args.dry_run == "True"
    sampling_method = args.sampling
    
    # Build parameter string for saving
    para = (f"smiles_{args.bulk}_bottle_{args.bottleneck}_edim_{args.encoder_h_dims}"
            f"_pdim_{args.predictor_h_dims}_model_{reduce_model}"
            f"_dropout_{args.dropout}_lr_{args.lr}")
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Create directories
    for path in [args.log, args.bulk_model, args.bulk_encoder, 'save/ori_result', 'save/figures']:
        if not os.path.exists(path):
            os.makedirs(path)
    
    preditor_path = args.bulk_model + para
    bulk_encoder = args.bulk_encoder + para
    
    # Initialize logging
    out_path = log_path + now + "bulk_smiles.err"
    log_file = log_path + now + "bulk_smiles.log"
    out = open(out_path, "w")
    sys.stderr = out
    
    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    
    # ========================================================================
    # 1. Load SMILES data and build drug token mapping
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 0: Loading SMILES data")
    print("=" * 60)
    
    vocab, max_len, vocab_size, drug_tokens, matched_drugs = prepare_smiles_data(
        smiles_path, label_path
    )
    
    # ========================================================================
    # 2. Load and prepare bulk data
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Loading bulk expression and label data")
    print("=" * 60)
    
    data_r = pd.read_csv(data_path, index_col=0)
    label_r = pd.read_csv(label_path, index_col=0)
    
    if args.bulk == 'old':
        data_r = data_r.iloc[:805]
        label_r = label_r.iloc[:805]
    elif args.bulk == 'new':
        data_r = data_r.iloc[805:]
        label_r = label_r.iloc[805:]
    
    label_r = label_r.fillna(na)
    
    print(f"  Expression matrix: {data_r.shape}")
    print(f"  Label matrix: {label_r.shape}")
    
    # Optional: highly variable genes (dispersion threshold)
    if g_disperson is not None:
        hvg, adata = ut.highly_variable_genes(data_r, min_disp=g_disperson)
        data_r.columns = adata.var_names
        data_r = data_r.loc[:, hvg]
        print(f"  After HVG selection: {data_r.shape}")

    # Optional: top-K highly variable genes
    if args.use_hvg:
        n_orig = data_r.shape[1]
        hvg_mask, hvg_adata = ut.highly_variable_genes(data_r, n_top_genes=args.n_top_genes)
        data_r.columns = hvg_adata.var_names
        data_r = data_r.loc[:, hvg_mask]
        print(f"[HVG] Genes originales: {n_orig} → Genes seleccionados: {data_r.shape[1]}")
        os.makedirs('save', exist_ok=True)
        with open('save/hvg_genes.txt', 'w') as f:
            f.write('\n'.join(data_r.columns.tolist()))
        print(f"[HVG] input_dim del modelo: {data_r.shape[1]}")
        print(f"[HVG] Lista de genes guardada en: save/hvg_genes.txt")
    
    # ========================================================================
    # 3. Build multi-drug dataset
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Building multi-drug dataset")
    print("=" * 60)
    
    X_gene, X_drug, Y = build_multidrug_dataset(
        data_r, label_r, drug_tokens, matched_drugs, 
        na_value=na, dry_run=dry_run
    )
    
    # Scale gene expression
    mmscaler = preprocessing.MinMaxScaler()
    X_gene = mmscaler.fit_transform(X_gene)
    
    # Labels are already encoded as 0 (sensitive) / 1 (resistant)
    dim_model_out = 2
    
    # Split train/valid/test
    X_gene_train_all, X_gene_test, X_drug_train_all, X_drug_test, Y_train_all, Y_test = \
        train_test_split(X_gene, X_drug, Y, test_size=test_size, random_state=42)
    
    X_gene_train, X_gene_valid, X_drug_train, X_drug_valid, Y_train, Y_valid = \
        train_test_split(X_gene_train_all, X_drug_train_all, Y_train_all, 
                         test_size=valid_size, random_state=42)
    
    # ---- Sampling (class balancing) ----
    if sampling_method != "no":
        print(f"\n  Applying {sampling_method} to balance classes...")
        print(f"  Before: sensitive(0)={np.sum(Y_train==0):,}, resistant(1)={np.sum(Y_train==1):,}")
        
        # Concatenate gene+drug for resampling, then split back
        n_gene_cols = X_gene_train.shape[1]
        X_combined = np.hstack([X_gene_train, X_drug_train.astype(np.float32)])
        
        if sampling_method == "upsampling":
            X_combined, Y_train = sam.upsampling(X_combined, Y_train)
        elif sampling_method == "downsampling":
            X_combined, Y_train = sam.downsampling(X_combined, Y_train)
        elif sampling_method == "SMOTE":
            X_combined, Y_train = sam.SMOTEsampling(X_combined, Y_train)
        
        # Split back into gene and drug
        X_gene_train = X_combined[:, :n_gene_cols].astype(np.float32)
        X_drug_train = X_combined[:, n_gene_cols:].astype(np.int64)
        
        print(f"  After:  sensitive(0)={np.sum(Y_train==0):,}, resistant(1)={np.sum(Y_train==1):,}")
        print(f"  New train size: {X_gene_train.shape[0]:,} samples")
    
    print(f"\n  Train: {X_gene_train.shape[0]:,} samples")
    print(f"  Valid: {X_gene_valid.shape[0]:,} samples")
    print(f"  Test:  {X_gene_test.shape[0]:,} samples")
    
    # Device
    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print(f"  Device: {device}")
    
    # ========================================================================
    # 4. Create tensors and data loaders
    # ========================================================================
    
    # Tensors for DAE pre-training (gene expression only, self-supervised)
    X_trainTensor = torch.FloatTensor(X_gene_train).to(device)
    X_validTensor = torch.FloatTensor(X_gene_valid).to(device)
    X_testTensor = torch.FloatTensor(X_gene_test).to(device)
    
    train_dataset_ae = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset_ae = TensorDataset(X_validTensor, X_validTensor)
    
    X_trainDataLoader = DataLoader(dataset=train_dataset_ae, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset_ae, batch_size=batch_size, shuffle=True)
    dataloaders_pretrain = {'train': X_trainDataLoader, 'val': X_validDataLoader}
    
    # Tensors for predictor training (gene + drug + label)
    D_trainTensor = torch.LongTensor(X_drug_train).to(device)
    D_validTensor = torch.LongTensor(X_drug_valid).to(device)
    D_testTensor = torch.LongTensor(X_drug_test).to(device)
    
    Y_trainTensor = torch.LongTensor(Y_train).to(device)
    Y_validTensor = torch.LongTensor(Y_valid).to(device)
    
    train_dataset_pred = TensorDataset(X_trainTensor, D_trainTensor, Y_trainTensor)
    valid_dataset_pred = TensorDataset(X_validTensor, D_validTensor, Y_validTensor)
    
    trainDataLoader_p = DataLoader(dataset=train_dataset_pred, batch_size=batch_size, shuffle=True)
    validDataLoader_p = DataLoader(dataset=valid_dataset_pred, batch_size=batch_size, shuffle=True)
    dataloaders_train = {'train': trainDataLoader_p, 'val': validDataLoader_p}
    
    # ========================================================================
    # 5. Pre-train DAE encoder (Phase 1 - same as original)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Pre-training DAE encoder (gene expression)")
    print("=" * 60)
    
    n_genes = X_gene_train.shape[1]
    print(f"  Input dim (genes): {n_genes}")
    print(f"  Bottleneck dim: {dim_au_out}")
    print(f"  Encoder dims: {encoder_hdims}")
    
    if str(args.pretrain) != "False":
        if reduce_model == 'DAE':
            encoder = AEBase(input_dim=n_genes, latent_dim=dim_au_out,
                            h_dims=encoder_hdims, drop_out=args.dropout)
        elif reduce_model == 'AE':
            encoder = AEBase(input_dim=n_genes, latent_dim=dim_au_out,
                            h_dims=encoder_hdims, drop_out=args.dropout)
        
        encoder.to(device)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
        
        load = bulk_encoder if args.checkpoint != "False" else False
        
        if reduce_model == "DAE":
            encoder, loss_report_en = t.train_DAE_model(
                net=encoder, data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, loss_function=loss_function_e, load=load,
                n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=bulk_encoder)
        elif reduce_model == "AE":
            encoder, loss_report_en = t.train_AE_model(
                net=encoder, data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, loss_function=loss_function_e, load=load,
                n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=bulk_encoder)
        
        print("  DAE encoder pre-training complete!")
    
    # ========================================================================
    # 6. Train PretrainedPredictorWithDrug (Phase 2 - NEW)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Training multi-drug predictor with SMILES")
    print("=" * 60)
    
    model = PretrainedPredictorWithDrug(
        # AE params
        input_dim=n_genes, latent_dim=dim_au_out,
        h_dims=encoder_hdims, drop_out=args.dropout,
        pretrained_weights=bulk_encoder, freezed=bool(args.freeze_pretrain),
        # Drug encoder params
        vocab_size=vocab_size, max_smiles_len=max_len,
        drug_embed_dim=args.drug_embed_dim, drug_h_dim=args.drug_h_dim,
        drug_latent_dim=args.drug_latent_dim, drug_drop_out=args.dropout,
        # Predictor params
        hidden_dims_predictor=preditor_hdims,
        drop_out_predictor=args.dropout, output_dim=dim_model_out
    )
    
    model.to(device)
    print(f"  Model architecture:")
    print(f"    Gene encoder: {n_genes} → {encoder_hdims} → {dim_au_out}")
    print(f"    Drug encoder: vocab={vocab_size}, max_len={max_len} → {args.drug_latent_dim}")
    print(f"    Predictor input: {dim_au_out} + {args.drug_latent_dim} = {dim_au_out + args.drug_latent_dim}")
    print(f"    Predictor: {preditor_hdims} → {dim_model_out}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    
    load = True if args.checkpoint != "False" else False
    
    model, report = t.train_predictor_model_smiles(
        model, dataloaders_train,
        optimizer, loss_function, epochs, exp_lr_scheduler,
        load=load, save_path=preditor_path
    )
    
    print("  Multi-drug predictor training complete!")
    
    # ========================================================================
    # 7. Evaluate on test set
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Evaluation")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        dl_result = model(X_testTensor, D_testTensor).detach().cpu().numpy()
    
    lb_results = np.argmax(dl_result, axis=1)
    pb_results = dl_result[:, 1]
    
    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    
    try:
        ap_score = average_precision_score(Y_test, pb_results)
        auroc_score = roc_auc_score(Y_test, pb_results)
        report_df['auroc_score'] = auroc_score
        report_df['ap_score'] = ap_score
        print(f"  AUROC: {auroc_score:.4f}")
        print(f"  AP:    {ap_score:.4f}")
    except Exception as e:
        print(f"  Warning: could not compute AUROC/AP: {e}")
    
    print(f"  Accuracy: {report_dict['accuracy']:.4f}")
    print(f"  F1 (weighted): {report_dict['weighted avg']['f1-score']:.4f}")
    
    report_df.to_csv(f"save/logs/{reduce_model}_smiles_{now}_report.csv")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("bulk_model_smiles finished ✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scDEAL bulk model training with SMILES")
    
    # Data paths
    parser.add_argument('--data', type=str, default='data/ALL_expression.csv',
                        help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',
                        help='Path of the bulk RNA-Seq drug screening annotation')
    parser.add_argument('--smiles_file', type=str, default='data/smile_inchi.csv',
                        help='Path to smile_inchi.csv with drug SMILES')
    
    # Data processing
    parser.add_argument('--missing_value', type=int, default=1,
                        help='Value for missing entries in drug labels. Default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction. Default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Validation set fraction. Default: 0.2')
    parser.add_argument('--var_genes_disp', type=float, default=None,
                        help='HVG dispersion threshold. None = all genes. Default: None')
    parser.add_argument('--use_hvg', action='store_true',
                        help='Select top-K highly variable genes before training. Default: disabled')
    parser.add_argument('--n_top_genes', type=int, default=2000,
                        help='Number of top HVGs to select (used with --use_hvg). Default: 2000')
    parser.add_argument('--bulk', type=str, default='integrate',
                        help='Bulk database: integrate/old/new. Default: integrate')
    
    # Model architecture
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='Bottleneck (latent) dimension. Default: 32')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256",
                        help='Encoder hidden dims. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="128,64",
                        help='Predictor hidden dims. Default: 128,64')
    parser.add_argument('--dimreduce', type=str, default="DAE",
                        help='Encoder type: AE or DAE. Default: DAE')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate. Default: 0.3')
    
    # Drug encoder params
    parser.add_argument('--drug_embed_dim', type=int, default=32,
                        help='SMILES character embedding dimension. Default: 32')
    parser.add_argument('--drug_h_dim', type=int, default=128,
                        help='Drug encoder hidden dimension. Default: 128')
    parser.add_argument('--drug_latent_dim', type=int, default=32,
                        help='Drug encoder output dimension. Default: 32')
    
    # Training
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device: cpu or gpu. Default: cpu')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size. Default: 200')
    parser.add_argument('--pretrain', type=str, default="True",
                        help='Whether to pretrain encoder. Default: True')
    parser.add_argument('--freeze_pretrain', type=int, default=0,
                        help='Freeze pretrained encoder. 0/1. Default: 0')
    parser.add_argument('--checkpoint', type=str, default='False',
                        help='Load from checkpoint. Default: False')
    
    # Sampling
    parser.add_argument('--sampling', type=str, default='no',
                        help='Sampling method: no, upsampling, downsampling, SMOTE. Default: no')
    
    # Dry run
    parser.add_argument('--dry_run', type=str, default="False",
                        help='Quick test with 3 drugs and minimal data. Default: False')
    
    # Paths
    parser.add_argument('--bulk_model', '-p', type=str, default='save/bulk_pre/',
                        help='Path for trained predictor model')
    parser.add_argument('--bulk_encoder', '-e', type=str, default='save/bulk_encoder/',
                        help='Path for pre-trained encoder')
    parser.add_argument('--log', '-l', type=str, default='save/logs/',
                        help='Path for training log')
    
    warnings.filterwarnings("ignore")
    args, unknown = parser.parse_known_args()
    
    import matplotlib
    matplotlib.use('Agg')
    
    run_main(args)
