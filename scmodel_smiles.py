#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scmodel_smiles.py
==================
Transfer learning from bulk to single-cell with SMILES drug integration.

Differences from original scmodel.py:
  - Source (bulk) data loader includes SMILES tokens: (gene, drug, label)
  - Uses DaNNWithDrug for domain adaptation
  - Uses TargetModelWithDrug for final prediction
  - Can predict with any drug at inference time via --predict_drug

Usage:
  # Full training:
  python scmodel_smiles.py --epochs 500 --dimreduce DAE --device cpu

  # Dry-run (quick local test):
  python scmodel_smiles.py --epochs 2 --dimreduce DAE --device cpu --dry_run True

  # Predict specific drug on single cells:
  python scmodel_smiles.py --predict_drug CISPLATIN --epochs 500 --dimreduce DAE
"""

import argparse
import pandas as pd
from pandas.core.frame import DataFrame
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
import utils as ut
from captum.attr import IntegratedGradients
from models import (AEBase, DaNNWithDrug, PretrainedPredictorWithDrug,
                    TargetModelWithDrug)
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

DATA_MAP = {
    "GSE117872": "data/GSE117872/GSE117872_good_Data_TPM.txt",
    "GSE110894": "data/GSE110894/GSE110894.csv",
    "GSE112274": "data/GSE112274/GSE112274_cell_gene_FPKM.csv",
    "GSE140440": "data/GSE140440/GSE140440.csv",
    "GSE149383": "data/GSE149383/erl_total_data_2K.csv",
    "GSE110894_small": "data/GSE110894/GSE110894_small.h5ad"
}


def build_source_multidrug_dataset(data_r, label_r, drug_tokens, matched_drugs,
                                    na_value=1, dry_run=False):
    """
    Build expanded source (bulk) dataset for transfer learning.
    Same logic as in bulkmodel_smiles.py.
    Labels are encoded as: sensitive → 0, resistant → 1
    """
    if dry_run:
        matched_drugs = matched_drugs[:3]
        print(f"[DRY-RUN] Using only {len(matched_drugs)} drugs: {matched_drugs}")
    
    # Align indices between expression and label data
    common_idx = data_r.index.intersection(label_r.index)
    data_r = data_r.loc[common_idx]
    label_r = label_r.loc[common_idx]
    
    # Label encoding map (must match bulk model convention: sensitive=0, resistant=1)\n    # The prediction section handles the flip to SC convention (sensitive=1, resistant=0)\n    label_map = {'sensitive': 0, 'resistant': 1}
    
    gene_rows = []
    drug_rows = []
    label_rows = []
    
    for drug_name in matched_drugs:
        if drug_name not in label_r.columns:
            continue
        
        drug_labels = label_r[drug_name]
        label_map = {'sensitive': 1, 'resistant': 0}
        valid_mask = drug_labels.isin(label_map.keys())
        valid_indices = valid_mask[valid_mask].index
        
        if len(valid_indices) == 0:
            continue
        
        gene_data = data_r.loc[valid_indices].values
        labels = drug_labels.loc[valid_indices].map(label_map).values
        tokens = drug_tokens[drug_name].numpy()
        drug_tokens_repeated = np.tile(tokens, (len(valid_indices), 1))
        
        gene_rows.append(gene_data)
        drug_rows.append(drug_tokens_repeated)
        label_rows.append(labels)
    
    X_gene = np.vstack(gene_rows).astype(np.float32)
    X_drug = np.vstack(drug_rows).astype(np.int64)
    Y = np.concatenate(label_rows).astype(np.int64)
    
    print(f"[SOURCE DATASET] Built multi-drug source dataset:")
    print(f"  Samples: {X_gene.shape[0]:,}, Genes: {X_gene.shape[1]:,}")
    print(f"  Drugs: {len(matched_drugs)}")
    print(f"  Label distribution: sensitive(0)={np.sum(Y==0):,}, resistant(1)={np.sum(Y==1):,}")
    
    return X_gene, X_drug, Y


def run_main(args):
    t0 = time.time()

    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck
    na = args.missing_value
    data_name = args.sc_data
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = list(map(int, args.bulk_h_dims.split(",")))
    predict_hdims = list(map(int, args.predictor_h_dims.split(",")))
    reduce_model = args.dimreduce
    leiden_res = args.cluster_res
    mod = args.mod
    dry_run = args.dry_run == "True"
    predict_drug = args.predict_drug
    
    # Resolve data path
    if args.sc_data in DATA_MAP:
        data_path = DATA_MAP[args.sc_data]
    else:
        data_path = args.sc_data
    
    source_data_path = args.bulk_data
    label_path = args.label
    smiles_path = args.smiles_file
    
    # Build parameter string for saving
    para = (f"smiles_{args.bulk}_data_{args.sc_data}_bottle_{dim_au_out}"
            f"_edim_{args.bulk_h_dims}_pdim_{args.predictor_h_dims}"
            f"_model_{reduce_model}_dropout_{args.dropout}"
            f"_lr_{args.lr}_mod_{mod}")
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Initialize logging
    out_path = log_path + now + "transfer_smiles.err"
    log_file = log_path + now + "transfer_smiles.log"
    
    # Create directories
    for path in [args.logging_file, args.bulk_model_path, args.sc_model_path,
                 args.sc_encoder_path, "save/adata/"]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    out = open(out_path, "w")
    sys.stderr = out
    
    logging.basicConfig(level=logging.INFO,
                        filename=log_file, filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    
    sc_encoder_path = args.sc_encoder_path + para
    source_model_path = args.bulk_model_path + para
    target_model_path = args.sc_model_path + para
    
    # ========================================================================
    # 1. Load SMILES data
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 0: Loading SMILES data")
    print("=" * 60)
    
    vocab, max_len, vocab_size, drug_tokens, matched_drugs = prepare_smiles_data(
        smiles_path, label_path
    )
    
    # ========================================================================
    # 2. Load and preprocess single-cell data (same as original)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Loading and preprocessing single-cell data")
    print("=" * 60)
    
    adata = pp.read_sc_file(data_path)
    
    if data_name == 'GSE117872_HN137':
        adata = ut.specific_process(adata, dataname='GSE117872', select_origin='HN137')
    elif data_name == 'GSE117872_HN120':
        adata = ut.specific_process(adata, dataname='GSE117872', select_origin='HN120')
    elif data_name in ['GSE122843', 'GSE110894', 'GSE112274', 'GSE116237',
                       'GSE108383', 'GSE140440', 'GSE129730', 'GSE149383']:
        adata = ut.specific_process(adata, dataname=data_name)
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = pp.cal_ncount_ngenes(adata)
    
    if data_name not in ['GSE112274', 'GSE140440']:
        adata = pp.receipe_my(adata, l_n_genes=min_n_genes, r_n_genes=max_n_genes,
                              filter_mincells=args.min_c, filter_mingenes=args.min_g,
                              normalize=True, log=True)
    else:
        adata = pp.receipe_my(adata, l_n_genes=min_n_genes, r_n_genes=max_n_genes,
                              filter_mincells=args.min_c, percent_mito=args.percent_mito,
                              filter_mingenes=args.min_g, normalize=True, log=True)
    
    sc.pp.highly_variable_genes(adata, min_disp=g_disperson, max_disp=np.inf, max_mean=6)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    data = adata.X
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.leiden(adata, resolution=leiden_res)
    sc.tl.umap(adata)
    adata.obs['leiden_origin'] = adata.obs['leiden']
    adata.obsm['X_umap_origin'] = adata.obsm['X_umap']
    data_c = adata.obs['leiden'].astype("long").to_list()
    
    print(f"  SC data shape: {data.shape}")
    print(f"  Number of cells: {adata.n_obs}")
    print(f"  Number of HVGs: {adata.n_vars}")
    
    # Normalize SC data
    mmscaler = preprocessing.MinMaxScaler()
    try:
        data = mmscaler.fit_transform(data)
    except:
        data = data.todense()
        data = mmscaler.fit_transform(data)
    
    # Split SC data
    Xtarget_train, Xtarget_valid, Ctarget_train, Ctarget_valid = \
        train_test_split(data, data_c, test_size=valid_size, random_state=42)
    
    # Device
    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print(f"  Device: {device}")
    
    # SC tensors on GPU for DAE pre-training (small, fits in memory)
    Xtarget_trainTensor_gpu = torch.FloatTensor(Xtarget_train).to(device)
    Xtarget_validTensor_gpu = torch.FloatTensor(Xtarget_valid).to(device)
    Ctarget_trainTensor_gpu = torch.LongTensor(Ctarget_train).to(device)
    Ctarget_validTensor_gpu = torch.LongTensor(Ctarget_valid).to(device)
    X_allTensor = torch.FloatTensor(data).to(device)
    
    # DAE pre-training loaders (GPU - used by original train_DAE_model)
    train_dataset_dae = TensorDataset(Xtarget_trainTensor_gpu, Xtarget_trainTensor_gpu)
    valid_dataset_dae = TensorDataset(Xtarget_validTensor_gpu, Xtarget_validTensor_gpu)
    dataloaders_pretrain = {'train': DataLoader(dataset=train_dataset_dae, batch_size=batch_size, shuffle=True),
                            'val': DataLoader(dataset=valid_dataset_dae, batch_size=batch_size, shuffle=True)}
    
    # DaNN transfer loaders (CPU - trainer moves batches to GPU)
    Xtarget_trainTensor_cpu = torch.FloatTensor(Xtarget_train)
    Xtarget_validTensor_cpu = torch.FloatTensor(Xtarget_valid)
    Ctarget_trainTensor_cpu = torch.LongTensor(Ctarget_train)
    Ctarget_validTensor_cpu = torch.LongTensor(Ctarget_valid)
    train_dataset_dann = TensorDataset(Xtarget_trainTensor_cpu, Ctarget_trainTensor_cpu)
    valid_dataset_dann = TensorDataset(Xtarget_validTensor_cpu, Ctarget_validTensor_cpu)
    dataloaders_target_dann = {'train': DataLoader(dataset=train_dataset_dann, batch_size=batch_size, shuffle=True),
                               'val': DataLoader(dataset=valid_dataset_dann, batch_size=batch_size, shuffle=True)}
    
    # ========================================================================
    # 3. Load and prepare bulk (source) data with SMILES
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Loading bulk source data with SMILES")
    print("=" * 60)
    
    data_r = pd.read_csv(source_data_path, index_col=0)
    label_r = pd.read_csv(label_path, index_col=0)
    
    if args.bulk == 'old':
        data_r = data_r.iloc[:805]
        label_r = label_r.iloc[:805]
    elif args.bulk == 'new':
        data_r = data_r.iloc[805:]
        label_r = label_r.iloc[805:]
    
    label_r = label_r.fillna(na)

    # Optional: filter bulk source to HVG genes used during bulk training
    if args.use_hvg and args.hvg_genes_path:
        with open(args.hvg_genes_path) as f:
            hvg_genes = f.read().splitlines()
        genes_missing = [g for g in hvg_genes if g not in data_r.columns]
        if genes_missing:
            print(f"[HVG] {len(genes_missing)} genes not found in bulk source, filling with 0")
        data_r = data_r.reindex(columns=hvg_genes, fill_value=0.0)
        print(f"[HVG] Bulk source filtrado a {len(hvg_genes)} genes del bulk entrenado")

    X_src_gene, X_src_drug, Y_src = build_source_multidrug_dataset(
        data_r, label_r, drug_tokens, matched_drugs,
        na_value=na, dry_run=dry_run
    )
    
    # Scale source data
    mmscaler_src = preprocessing.MinMaxScaler()
    X_src_gene = mmscaler_src.fit_transform(X_src_gene)
    
    # Labels are already encoded as 0/1
    dim_model_out = 2
    
    # Split source data
    Xsrc_gene_train, Xsrc_gene_valid, Xsrc_drug_train, Xsrc_drug_valid, \
        Ysrc_train, Ysrc_valid = train_test_split(
            X_src_gene, X_src_drug, Y_src, test_size=valid_size, random_state=42)
    
    # Source tensors - keep on CPU, batches move to GPU in trainer
    Xsrc_gene_trainTensor = torch.FloatTensor(Xsrc_gene_train)
    Xsrc_gene_validTensor = torch.FloatTensor(Xsrc_gene_valid)
    Xsrc_drug_trainTensor = torch.LongTensor(Xsrc_drug_train)
    Xsrc_drug_validTensor = torch.LongTensor(Xsrc_drug_valid)
    Ysrc_trainTensor = torch.LongTensor(Ysrc_train)
    Ysrc_validTensor = torch.LongTensor(Ysrc_valid)
    
    source_train_dataset = TensorDataset(Xsrc_gene_trainTensor, Xsrc_drug_trainTensor, Ysrc_trainTensor)
    source_valid_dataset = TensorDataset(Xsrc_gene_validTensor, Xsrc_drug_validTensor, Ysrc_validTensor)
    
    Xsource_trainDataLoader = DataLoader(dataset=source_train_dataset, batch_size=batch_size, shuffle=True)
    Xsource_validDataLoader = DataLoader(dataset=source_valid_dataset, batch_size=batch_size, shuffle=True)
    dataloaders_source = {'train': Xsource_trainDataLoader, 'val': Xsource_validDataLoader}
    
    # ========================================================================
    # 4. Build models
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Building models")
    print("=" * 60)
    
    n_genes_sc = data.shape[1]
    n_genes_bulk = X_src_gene.shape[1]
    
    # Target (SC) encoder
    if reduce_model in ["AE", "DAE"]:
        encoder = AEBase(input_dim=n_genes_sc, latent_dim=dim_au_out,
                         h_dims=encoder_hdims, drop_out=args.dropout)
        loss_function_e = nn.MSELoss()
    
    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
    
    # Source (bulk) model with SMILES
    source_model = PretrainedPredictorWithDrug(
        input_dim=n_genes_bulk, latent_dim=dim_au_out,
        h_dims=encoder_hdims, drop_out=args.dropout,
        pretrained_weights=None, freezed=False,
        vocab_size=vocab_size, max_smiles_len=max_len,
        drug_embed_dim=args.drug_embed_dim, drug_h_dim=args.drug_h_dim,
        drug_latent_dim=args.drug_latent_dim, drug_drop_out=args.dropout,
        hidden_dims_predictor=predict_hdims,
        drop_out_predictor=args.dropout, output_dim=dim_model_out
    )
    
    # Load pre-trained bulk model
    bulk_model_file = args.bulk_model_file if args.bulk_model_file else source_model_path
    
    if os.path.exists(bulk_model_file):
        source_model.load_state_dict(torch.load(bulk_model_file, map_location=device))
        print(f"  ✅ Loaded pre-trained bulk model from: {bulk_model_file}")
    else:
        print(f"  ⚠️ WARNING: Pre-trained bulk model not found at: {bulk_model_file}")
        print(f"  Available models in save/bulk_pre/:")
        bulk_dir = 'save/bulk_pre/'
        if os.path.exists(bulk_dir):
            for f in os.listdir(bulk_dir):
                if 'smiles' in f:
                    print(f"    → {bulk_dir}{f}")
        print(f"  Use --bulk_model_file <path> to specify the correct file.")
        print(f"  Training will proceed without pre-trained weights.")
    
    source_model.to(device)
    
    print(f"  SC encoder input: {n_genes_sc} genes")
    print(f"  Bulk model input: {n_genes_bulk} genes")
    print(f"  Bottleneck dim: {dim_au_out}")
    
    # ========================================================================
    # 5. Pre-train SC encoder (same as original)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Pre-training SC encoder (DAE)")
    print("=" * 60)
    
    if str(args.sc_encoder_path) != 'False':
        train_flag = True
        
        if args.checkpoint != "False":
            try:
                encoder.load_state_dict(torch.load(sc_encoder_path, map_location=device))
                print(f"  Loaded pre-trained SC encoder from: {sc_encoder_path}")
                train_flag = False
            except:
                print("  Loading failed, will re-train SC encoder")
                train_flag = True
        
        if train_flag:
            if reduce_model == "DAE":
                encoder, loss_report_en = t.train_DAE_model(
                    net=encoder, data_loaders=dataloaders_pretrain,
                    optimizer=optimizer_e, loss_function=loss_function_e, load=False,
                    n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=sc_encoder_path)
            elif reduce_model == "AE":
                encoder, loss_report_en = t.train_AE_model(
                    net=encoder, data_loaders=dataloaders_pretrain,
                    optimizer=optimizer_e, loss_function=loss_function_e, load=False,
                    n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=sc_encoder_path)
            print("  SC encoder pre-training complete!")
    
    # ========================================================================
    # 6. Transfer learning with DaNNWithDrug
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Transfer learning (DaNN with SMILES)")
    print("=" * 60)
    
    # Free GPU SC data used by DAE pre-training (no longer needed)
    del Xtarget_trainTensor_gpu, Xtarget_validTensor_gpu
    del Ctarget_trainTensor_gpu, Ctarget_validTensor_gpu
    del dataloaders_pretrain, train_dataset_dae, valid_dataset_dae
    torch.cuda.empty_cache()
    
    loss_d = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(encoder.parameters(), lr=args.lr)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)
    print(f"  DaNN optimizer LR: {args.lr}")
    
    DaNN_model = DaNNWithDrug(
        source_model=source_model, target_model=encoder,
        fix_source=bool(args.fix_source)
    )
    DaNN_model.to(device)
    
    def loss(x, y, GAMMA=args.mmd_GAMMA):
        return mmd.mmd_loss(x, y, GAMMA)
    
    loss_distribution = loss
    
    print(f"  Transfer learning mode: {mod}")
    print(f"  MMD weight: {args.mmd_weight}")
    
    if mod == 'new':
        DaNN_model, report_, _, _ = t.train_DaNN_model_smiles(
            DaNN_model,
            dataloaders_source, dataloaders_target_dann,
            optimizer_d, loss_d,
            epochs, exp_lr_scheduler_d,
            dist_loss=loss_distribution,
            load=False,
            weight=args.mmd_weight,
            save_path=target_model_path + "_DaNN.pkl",
            device=device)
    else:
        # For 'ori' mode, use simpler DaNN (without cluster regularization)
        DaNN_model, report_, _, _ = t.train_DaNN_model_smiles(
            DaNN_model,
            dataloaders_source, dataloaders_target_dann,
            optimizer_d, loss_d,
            epochs, exp_lr_scheduler_d,
            dist_loss=loss_distribution,
            load=False,
            weight=args.mmd_weight,
            save_path=target_model_path + "_DaNN.pkl",
            device=device)
    
    encoder = DaNN_model.target_model
    source_model = DaNN_model.source_model
    print("  Transfer learning complete!")
    
    # ========================================================================
    # 7. Prediction
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Prediction on single cells")
    print("=" * 60)
    
    # Get prediction drug tokens
    if predict_drug and predict_drug in drug_tokens:
        pred_drug_name = predict_drug
    elif predict_drug:
        # Try to find it in matched_drugs
        pred_drug_name = None
        predict_upper = predict_drug.upper().replace('-', '.').replace(' ', '.')
        for md in matched_drugs:
            if md.upper() == predict_upper:
                pred_drug_name = md
                break
        if pred_drug_name is None:
            print(f"  WARNING: Drug '{predict_drug}' not found. Using first drug: {matched_drugs[0]}")
            pred_drug_name = matched_drugs[0]
    else:
        pred_drug_name = matched_drugs[0]
    
    print(f"  Predicting drug: {pred_drug_name}")
    
    # Create drug token tensor for all cells (same drug for all)
    n_cells = X_allTensor.shape[0]
    drug_token_vec = drug_tokens[pred_drug_name].to(device)
    drug_tokens_all = drug_token_vec.unsqueeze(0).expand(n_cells, -1)
    
    # Get embeddings and predictions
    embedding_tensors = encoder.encode(X_allTensor)
    drug_emb_tensors = source_model.drug_encoder(drug_tokens_all)
    prediction_tensors = source_model.predictor(embedding_tensors, drug_emb_tensors)
    
    embeddings = embedding_tensors.detach().cpu().numpy()
    predictions = prediction_tensors.detach().cpu().numpy()
    
    print(f"  Predictions shape: {predictions.shape}")
    
    # Store results (SC convention: sensitive=1, resistant=0)
    # Bulk model: class 0=sensitive, class 1=resistant
    # SC ground truth: 1=sensitive, 0=resistant
    # So: P(sensitive_SC=1) = P(bulk_class_0) = predictions[:, 0]
    adata.obs["sens_preds"] = predictions[:, 0]  # P(sensitive) in SC convention
    adata.obs["rest_preds"] = predictions[:, 1]  # P(resistant) in SC convention
    adata.obs["sens_label"] = (1 - predictions.argmax(axis=1))  # Flip: bulk 0→SC 1, bulk 1→SC 0
    adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
    adata.obs["predict_drug"] = pred_drug_name
    adata.obsm["X_transfer"] = embeddings
    
    # Save adata
    save_name = f"save/adata/{data_name}_{para}_smiles.h5ad"
    adata.write(save_name)
    print(f"  Saved results to: {save_name}")
    
    # ========================================================================
    # 8. Evaluate if ground truth is available
    # ========================================================================
    if 'sensitive' in adata.obs.columns:
        from sklearn.metrics import (average_precision_score,
                                     classification_report, roc_auc_score)
        Y_test = adata.obs['sensitive']
        sens_pb_results = adata.obs['sens_preds']
        lb_results = adata.obs['sens_label']
        
        try:
            report_dict = classification_report(Y_test, lb_results, output_dict=True)
            f1score = report_dict['weighted avg']['f1-score']
            print(f"\n  Evaluation against ground truth:")
            print(f"    F1 (weighted): {f1score:.4f}")
            
            ap_score = average_precision_score(Y_test, sens_pb_results)
            print(f"    AP: {ap_score:.4f}")
        except Exception as e:
            print(f"  Could not compute metrics: {e}")
    
    # Summary
    print(f"\n  Prediction summary for {pred_drug_name}:")
    print(f"    Sensitive: {(adata.obs['sens_label'] == 0).sum()}")
    print(f"    Resistant: {(adata.obs['sens_label'] == 1).sum()}")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print("scmodel_smiles finished ✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scDEAL transfer learning with SMILES")
    
    # Data paths
    parser.add_argument('--bulk_data', type=str, default='data/ALL_expression.csv',
                        help='Path of bulk expression data')
    parser.add_argument('--label', type=str, default='data/ALL_label_binary_wf.csv',
                        help='Path of bulk drug labels')
    parser.add_argument('--smiles_file', type=str, default='data/smile_inchi.csv',
                        help='Path to smile_inchi.csv')
    parser.add_argument('--sc_data', type=str, default="GSE110894",
                        help='SC dataset accession or path')
    parser.add_argument('--predict_drug', type=str, default='I.BET.762',
                        help='Drug to predict on single cells. Default: I.BET.762')
    
    # Data processing
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=0)
    parser.add_argument('--use_hvg', action='store_true',
                        help='Filter bulk source data to HVG genes from bulk training. Default: disabled')
    parser.add_argument('--hvg_genes_path', type=str, default=None,
                        help='Path to hvg_genes.txt saved during bulk training. Default: None')
    parser.add_argument('--min_n_genes', type=int, default=0)
    parser.add_argument('--max_n_genes', type=int, default=20000)
    parser.add_argument('--min_g', type=int, default=200)
    parser.add_argument('--min_c', type=int, default=3)
    parser.add_argument('--percent_mito', type=int, default=100)
    parser.add_argument('--cluster_res', type=float, default=0.2)
    parser.add_argument('--bulk', type=str, default='integrate')
    
    # MMD params
    parser.add_argument('--mmd_weight', type=float, default=0.25)
    parser.add_argument('--mmd_GAMMA', type=int, default=1000)
    
    # Model architecture
    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--bulk_h_dims', type=str, default="512,256")
    parser.add_argument('--sc_h_dims', type=str, default="512,256")
    parser.add_argument('--predictor_h_dims', type=str, default="128,64")
    parser.add_argument('--dimreduce', type=str, default="DAE")
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Drug encoder params
    parser.add_argument('--drug_embed_dim', type=int, default=32)
    parser.add_argument('--drug_h_dim', type=int, default=128)
    parser.add_argument('--drug_latent_dim', type=int, default=32)
    
    # Training
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--fix_source', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='False')
    parser.add_argument('--mod', type=str, default='new')
    parser.add_argument('--printgene', type=str, default='F')
    
    # Dry run
    parser.add_argument('--dry_run', type=str, default="False")
    
    # Bulk model file (direct path)
    parser.add_argument('--bulk_model_file', type=str, default=None,
                        help='Direct path to pre-trained bulk model .pkl file from bulkmodel_smiles.py')
    
    # Paths
    parser.add_argument('--bulk_model_path', '-s', type=str, default='save/bulk_pre/')
    parser.add_argument('--sc_model_path', '-p', type=str, default='save/sc_pre/')
    parser.add_argument('--sc_encoder_path', type=str, default='save/sc_encoder/')
    parser.add_argument('--logging_file', '-l', type=str, default='save/logs/')
    
    args, unknown = parser.parse_known_args()
    run_main(args)
