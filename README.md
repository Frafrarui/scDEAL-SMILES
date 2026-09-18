# scDEAL + SMILES

Fork of [scDEAL](https://github.com/OSU-BMBL/scDEAL) (*Deep Transfer Learning of
Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data*) that adds a
**SMILES-based multi-drug model**: instead of training one independent model per
drug, the drug's chemical structure is encoded and fed into the model alongside
the gene expression profile, enabling a single model to predict sensitivity for
any drug with an available SMILES — including drugs never seen during training.

For the original pipeline (`bulkmodel.py` / `scmodel.py`), installation,
conda environment, full dataset downloads and demo instructions, see the
**[original scDEAL repository and README](https://github.com/OSU-BMBL/scDEAL)**.
This document only covers what this fork adds on top of it.

## What changed vs. the original scDEAL

- **New input**: each drug's SMILES string is character-tokenized (see
  `smiles_encoder.py`) into a fixed-length integer vector and embedded by a new
  `DrugEncoder` module (`models.py`).
- **New models** (`models.py`): `DrugEncoder`, `PretrainedPredictorWithDrug`,
  `DaNNWithDrug`, `TargetModelWithDrug`. These mirror the original
  `PretrainedPredictor` / `DaNN` / `TargetModel` classes, but concatenate the
  gene embedding with the drug embedding before the predictor head.
- **New training scripts**:
  - `bulkmodel_smiles.py` — bulk-level pretraining on **all drugs at once**.
    Each training sample is `(gene_expression, smiles_tokens, label)`; the
    dataset expands from ~1,280 cell lines to ~106K (cell × drug) pairs.
  - `scmodel_smiles.py` — transfer learning + single-cell prediction. The
    single-cell encoder is trained independently from the bulk encoder (no
    forced gene-space correspondence between domains), and at inference time
    you can predict sensitivity for **any drug with an available SMILES**,
    including drugs never seen during training (`--predict_drug`).
  - `trainers.py` gained `train_predictor_model_smiles` and
    `train_DaNN_model_smiles`, the SMILES-aware counterparts of the original
    training loops.
- **New data file**: `data/smile_inchi.csv` — drug name → SMILES/InChI lookup
  table (sourced from DeepTTA, completed with 10 manually-curated entries
  tagged `User_Added` in the `Sample Size` column for drugs missing from the
  original GDSC screens).
- **Class imbalance handling**: `--sampling` now also supports `SMOTE`, via
  `sampling.py` (unchanged from the original scDEAL utility).
- `--use_hvg` (both scripts) optionally restricts bulk training to the top-N
  highly variable genes (`seurat_v3` flavor); the selected gene list is saved
  to `save/hvg_genes.txt` so the same gene subset can be reused when loading
  the bulk source encoder during single-cell transfer (`--hvg_genes_path`).
- `--fix_source` (`scmodel_smiles.py`) freezes the pretrained bulk
  predictor's weights during domain adaptation, so only the single-cell
  encoder is updated — this avoids catastrophic forgetting of the bulk-level
  drug-response knowledge.

## Required data

Same base files as the original pipeline (`data/ALL_expression.csv`,
`data/ALL_label_binary_wf.csv`, see the [original README](https://github.com/OSU-BMBL/scDEAL)
for where to download them), plus:

- `data/smile_inchi.csv` — SMILES lookup table (see above).

## Usage

**1. Bulk multi-drug pretraining** (`bulkmodel_smiles.py`):

```
# Quick local smoke test (~30s, 2 epochs)
python bulkmodel_smiles.py --epochs 2 --dimreduce DAE --device cpu --dry_run True

# Full training
python bulkmodel_smiles.py --epochs 500 --lr 0.001 --dimreduce DAE --device cpu

# With highly-variable-gene selection and upsampling for class imbalance
python bulkmodel_smiles.py --epochs 500 --lr 0.001 --use_hvg --n_top_genes 2000 \
    --sampling upsampling --device cpu
```

This trains `PretrainedPredictorWithDrug` jointly on all 83 drugs with
available SMILES, saving the bulk encoder to `save/bulk_encoder/` and the
predictor to `save/bulk_pre/`.

**2. Single-cell transfer + prediction** (`scmodel_smiles.py`):

```
# Transfer to a built-in scRNA-seq dataset, freezing the bulk predictor
python scmodel_smiles.py --sc_data GSE110894 --epochs 500 --lr 0.001 \
    --fix_source 1 --mmd_weight 0.25 --device cpu

# Predict sensitivity to a drug not necessarily tied to that dataset's
# original drug (zero-shot across drugs)
python scmodel_smiles.py --sc_data GSE117872 --predict_drug GEFITINIB \
    --epochs 500 --device cpu
```

Built-in `--sc_data` accessions (see `DATA_MAP` in `scmodel_smiles.py`):
`GSE110894`, `GSE117872`, `GSE112274`, `GSE140440`, `GSE149383`.

## Key arguments (SMILES scripts)

These are in addition to the arguments shared with `bulkmodel.py` / `scmodel.py`
(`--lr`, `--epochs`, `--bottleneck`, `--dropout`, `--encoder_h_dims` /
`--bulk_h_dims`, `--predictor_h_dims`, `--dimreduce`, `--device`,
`--checkpoint`, `--sampling`, `--var_genes_disp`, `--min_g`, `--min_c`,
`--mmd_weight`, `--mmd_GAMMA`, `--cluster_res`, etc. — see the original README
for those):

| Argument | Script(s) | Default | Description |
|---|---|---|---|
| `--smiles_file` | both | `data/smile_inchi.csv` | Path to the SMILES lookup table |
| `--drug_embed_dim` | both | 32 | Character-embedding dimension inside `DrugEncoder` |
| `--drug_h_dim` | both | 128 | Hidden layer size of `DrugEncoder`'s MLP |
| `--drug_latent_dim` | both | 32 | Output dimension of the drug embedding (concatenated with the gene embedding) |
| `--use_hvg` | both | off (flag) | Restrict bulk training to the top `--n_top_genes` highly variable genes |
| `--n_top_genes` | `bulkmodel_smiles.py` | 2000 | Number of HVGs to select when `--use_hvg` is set |
| `--hvg_genes_path` | `scmodel_smiles.py` | `None` | Path to the `hvg_genes.txt` saved during bulk training, to reuse the same gene subset |
| `--dry_run` | both | `False` | Quick smoke-test mode with a reduced dataset |
| `--predict_drug` | `scmodel_smiles.py` | `I.BET.762` | Drug (label column name) to predict sensitivity for on the single-cell data; must have a SMILES entry in `--smiles_file`, need not match the dataset's original drug |
| `--fix_source` | `scmodel_smiles.py` | 0 | Freeze the pretrained bulk predictor's weights during transfer (1) or update them jointly with the SC encoder (0) |
| `--sc_data` | `scmodel_smiles.py` | `GSE110894` | Built-in scRNA-seq accession (see `DATA_MAP`) or a custom path |

Run `python bulkmodel_smiles.py --help` / `python scmodel_smiles.py --help` for
the full, always-up-to-date argument list.
