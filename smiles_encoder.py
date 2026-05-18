#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smiles_encoder.py
=================
Utility module for encoding drug SMILES strings into numerical token tensors
for use with the scDEAL SMILES pipeline.

Character-level tokenization:
  Each unique character in the SMILES alphabet gets an integer index.
  Index 0 is reserved for padding.
  
Usage:
  from smiles_encoder import load_drug_smiles_map, build_smiles_vocab, tokenize_all_drugs

Functions:
  - load_drug_smiles_map(csv_path): Load drug Name → SMILES mapping from smile_inchi.csv
  - build_smiles_vocab(drug_smiles_dict): Build character vocabulary from SMILES
  - tokenize_smiles(smiles_str, vocab, max_len): Tokenize a single SMILES string
  - tokenize_all_drugs(drug_smiles_dict, vocab, max_len): Tokenize all drugs
  - get_drug_name_mapping(): Returns mapping from ALL_label_binary_wf.csv names to smile_inchi.csv names
"""

import csv
import numpy as np
import torch
import os


# ============================================================================
# Name normalization helpers
# ============================================================================

def _normalize_drug_name(name):
    """
    Normalize drug name for matching between ALL_label_binary_wf.csv and smile_inchi.csv.
    
    ALL_label_binary_wf.csv uses R-style names: dots instead of hyphens/spaces,
    and 'X' prefix for names starting with digits (e.g., X5.FLUOROURACIL).
    smile_inchi.csv uses standard names (e.g., 5-Fluorouracil).
    """
    name = name.strip().upper()
    # Remove R-style 'X' prefix for names starting with a digit
    if len(name) > 1 and name[0] == 'X' and name[1].isdigit():
        name = name[1:]
    # Normalize separators: replace dots with hyphens
    name = name.replace('.', '-')
    return name


def _normalize_smile_name(name):
    """Normalize drug name from smile_inchi.csv for matching."""
    return name.strip().upper().replace(' ', '-')


# ============================================================================
# Core functions
# ============================================================================

def load_drug_smiles_map(csv_path):
    """
    Load drug Name → SMILES mapping from smile_inchi.csv.
    
    Args:
        csv_path: Path to smile_inchi.csv
        
    Returns:
        dict: {normalized_drug_name: smiles_string}
              Names are uppercased with hyphens as separators.
    """
    drug_smiles = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_smile_name(row['Name'])
            smiles = row['smiles'].strip()
            if name not in drug_smiles and len(smiles) > 0:
                drug_smiles[name] = smiles
    return drug_smiles


def build_smiles_vocab(drug_smiles_dict):
    """
    Build character-level vocabulary from all SMILES strings.
    
    Args:
        drug_smiles_dict: dict {drug_name: smiles_string}
        
    Returns:
        vocab: dict {character: integer_index} (0 = padding)
        max_len: int, length of longest SMILES string
    """
    all_chars = set()
    max_len = 0
    for smiles in drug_smiles_dict.values():
        all_chars.update(smiles)
        max_len = max(max_len, len(smiles))
    
    # Sort for reproducibility, index 0 reserved for padding
    sorted_chars = sorted(all_chars)
    vocab = {ch: i + 1 for i, ch in enumerate(sorted_chars)}
    
    return vocab, max_len


def tokenize_smiles(smiles_str, vocab, max_len):
    """
    Tokenize a single SMILES string into a padded integer tensor.
    
    Args:
        smiles_str: SMILES string (e.g., "CCO")
        vocab: character → index mapping
        max_len: pad/truncate to this length
        
    Returns:
        torch.LongTensor of shape (max_len,)
    """
    tokens = [vocab.get(ch, 0) for ch in smiles_str[:max_len]]
    # Pad with 0s
    tokens += [0] * (max_len - len(tokens))
    return torch.LongTensor(tokens)


def tokenize_all_drugs(drug_smiles_dict, vocab, max_len):
    """
    Tokenize all drugs in the dictionary.
    
    Args:
        drug_smiles_dict: dict {normalized_drug_name: smiles_string}
        vocab: character → index mapping
        max_len: pad/truncate to this length
        
    Returns:
        dict: {normalized_drug_name: torch.LongTensor of shape (max_len,)}
    """
    tokens = {}
    for name, smiles in drug_smiles_dict.items():
        tokens[name] = tokenize_smiles(smiles, vocab, max_len)
    return tokens


def match_label_drugs_to_smiles(label_drug_names, drug_smiles_dict):
    """
    Match drug column names from ALL_label_binary_wf.csv to SMILES dictionary keys.
    
    Args:
        label_drug_names: list of drug names from ALL_label_binary_wf.csv columns
        drug_smiles_dict: dict from load_drug_smiles_map()
        
    Returns:
        matched: dict {label_column_name: normalized_smiles_key}
        unmatched: list of label column names with no match
    """
    matched = {}
    unmatched = []
    
    for label_name in label_drug_names:
        norm_label = _normalize_drug_name(label_name)
        
        if norm_label in drug_smiles_dict:
            matched[label_name] = norm_label
        else:
            # Try matching without hyphens
            norm_label_flat = norm_label.replace('-', '')
            found = False
            for smiles_key in drug_smiles_dict:
                if smiles_key.replace('-', '') == norm_label_flat:
                    matched[label_name] = smiles_key
                    found = True
                    break
            if not found:
                unmatched.append(label_name)
    
    return matched, unmatched


def prepare_smiles_data(smiles_csv_path, label_csv_path):
    """
    Full pipeline: load SMILES, build vocab, tokenize, and match to label drugs.
    
    This is the main entry point for the training scripts.
    
    Args:
        smiles_csv_path: path to smile_inchi.csv
        label_csv_path: path to ALL_label_binary_wf.csv
        
    Returns:
        vocab: dict {char: int}
        max_len: int
        vocab_size: int (including padding token)
        drug_tokens: dict {label_column_name: torch.LongTensor(max_len)}
        matched_drugs: list of matched drug column names
    """
    import pandas as pd
    
    # 1. Load SMILES
    drug_smiles = load_drug_smiles_map(smiles_csv_path)
    print(f"[SMILES] Loaded {len(drug_smiles)} unique drugs from {os.path.basename(smiles_csv_path)}")
    
    # 2. Build vocabulary
    vocab, max_len = build_smiles_vocab(drug_smiles)
    vocab_size = len(vocab) + 1  # +1 for padding token at index 0
    print(f"[SMILES] Vocabulary size: {vocab_size} (including padding)")
    print(f"[SMILES] Max SMILES length: {max_len}")
    
    # 3. Tokenize all drugs
    all_tokens = tokenize_all_drugs(drug_smiles, vocab, max_len)
    
    # 4. Match label drugs to SMILES
    label_df = pd.read_csv(label_csv_path, index_col=0, nrows=0)  # Only header
    label_drug_names = list(label_df.columns)
    
    matched, unmatched = match_label_drugs_to_smiles(label_drug_names, drug_smiles)
    print(f"[SMILES] Matched {len(matched)}/{len(label_drug_names)} drugs from labels")
    if unmatched:
        print(f"[SMILES] WARNING: Unmatched drugs: {unmatched}")
    
    # 5. Build final token dict: label_column_name → token tensor
    drug_tokens = {}
    for label_name, smiles_key in matched.items():
        drug_tokens[label_name] = all_tokens[smiles_key]
    
    return vocab, max_len, vocab_size, drug_tokens, list(matched.keys())


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SANITY CHECK: smiles_encoder.py")
    print("=" * 70)
    
    # Paths (adjust if running from a different directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smiles_csv = os.path.join(script_dir, 'data', 'smile_inchi.csv')
    label_csv = os.path.join(script_dir, 'data', 'ALL_label_binary_wf.csv')
    
    # Check files exist
    assert os.path.exists(smiles_csv), f"File not found: {smiles_csv}"
    assert os.path.exists(label_csv), f"File not found: {label_csv}"
    
    # Run full pipeline
    vocab, max_len, vocab_size, drug_tokens, matched_drugs = prepare_smiles_data(
        smiles_csv, label_csv
    )
    
    print(f"\n--- Results ---")
    print(f"Vocabulary size:    {vocab_size}")
    print(f"Max SMILES length:  {max_len}")
    print(f"Matched drugs:      {len(matched_drugs)}")
    print(f"Token tensor shape: {drug_tokens[matched_drugs[0]].shape}")
    
    # Show first 3 drugs
    print(f"\n--- Example tokens (first 3 drugs) ---")
    for drug in matched_drugs[:3]:
        t = drug_tokens[drug]
        print(f"  {drug:25s} → first 20 tokens: {t[:20].tolist()}")
    
    # Shape check
    for drug in matched_drugs:
        assert drug_tokens[drug].shape == (max_len,), \
            f"Shape mismatch for {drug}: {drug_tokens[drug].shape} != ({max_len},)"
    
    print(f"\n✅ All {len(matched_drugs)} drug tokens have correct shape ({max_len},)")
    print("✅ smiles_encoder.py sanity check PASSED")
