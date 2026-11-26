"""
MLP training on v14 with full metrics by drug order.
"""
import os
import sys
import ast
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# Load SapBERT for SE embeddings
print("Loading SapBERT...")
sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sapbert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sapbert_model.eval()

def get_sapbert_embedding(text):
    with torch.no_grad():
        inputs = sapbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = sapbert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

class DDIModel(nn.Module):
    def __init__(self, num_drugs, drug_emb_dim=128, se_emb_dim=768, hidden_dim=64):
        super().__init__()
        self.drug_embedding = nn.Embedding(num_drugs, drug_emb_dim)
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.se_projector = nn.Sequential(
            nn.Linear(se_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, drug_indices, drug_mask, se_features):
        drug_emb = self.drug_embedding(drug_indices)
        mask_expanded = drug_mask.unsqueeze(-1)
        drug_emb_sum = (drug_emb * mask_expanded).sum(dim=1)
        drug_count = drug_mask.sum(dim=1, keepdim=True).clamp(min=1)
        drug_emb_mean = drug_emb_sum / drug_count
        drug_proj = self.drug_projector(drug_emb_mean)
        se_proj = self.se_projector(se_features)
        return self.classifier(torch.cat([drug_proj, se_proj], dim=-1))

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'auprc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

def compute_ranking_metrics(se_indices, labels, probs):
    """Compute MRR, Avg AUROC/SE, Recall@K"""
    se_to_data = defaultdict(lambda: {'labels': [], 'probs': []})
    for se, label, prob in zip(se_indices, labels, probs):
        se_to_data[se]['labels'].append(label)
        se_to_data[se]['probs'].append(prob)

    mrr_sum, auroc_sum, valid_se = 0.0, 0.0, 0
    recall_at_10, recall_at_30, recall_at_50 = 0.0, 0.0, 0.0

    for se, data in se_to_data.items():
        y = np.array(data['labels'])
        p = np.array(data['probs'])

        if y.sum() == 0 or y.sum() == len(y):
            continue

        valid_se += 1

        # AUROC per SE
        auroc_sum += roc_auc_score(y, p)

        # Rank by prob descending
        sorted_idx = np.argsort(-p)
        sorted_labels = y[sorted_idx]

        # MRR: reciprocal rank of first positive
        pos_ranks = np.where(sorted_labels == 1)[0]
        if len(pos_ranks) > 0:
            mrr_sum += 1.0 / (pos_ranks[0] + 1)

        # Recall@K
        n_pos = y.sum()
        for k, recall_sum in [(10, 'r10'), (30, 'r30'), (50, 'r50')]:
            top_k = sorted_labels[:k]
            if recall_sum == 'r10':
                recall_at_10 += top_k.sum() / n_pos
            elif recall_sum == 'r30':
                recall_at_30 += top_k.sum() / n_pos
            else:
                recall_at_50 += top_k.sum() / n_pos

    return {
        'mrr': mrr_sum / valid_se if valid_se > 0 else 0.0,
        'avg_auroc_se': auroc_sum / valid_se if valid_se > 0 else 0.0,
        'recall@10': recall_at_10 / valid_se if valid_se > 0 else 0.0,
        'recall@30': recall_at_30 / valid_se if valid_se > 0 else 0.0,
        'recall@50': recall_at_50 / valid_se if valid_se > 0 else 0.0
    }

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    data_dir = 'dataset/HODDI_v15/HODDI/HGNN'
    pos_dir = os.path.join(data_dir, 'positive_samples')
    neg_dir = os.path.join(data_dir, 'negative_samples')

    pos_dfs, neg_dfs = [], []
    for f in sorted(os.listdir(pos_dir)):
        if f.endswith('.csv'):
            pos_dfs.append(pd.read_csv(os.path.join(pos_dir, f)))
    for f in sorted(os.listdir(neg_dir)):
        if f.endswith('.csv'):
            neg_dfs.append(pd.read_csv(os.path.join(neg_dir, f)))

    pos_df = pd.concat(pos_dfs, ignore_index=True)
    neg_df = pd.concat(neg_dfs, ignore_index=True)

    print(f"Loaded {len(pos_df)} positive and {len(neg_df)} negative samples")

    # Balance data
    min_samples = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=min_samples, random_state=42)
    neg_df = neg_df.sample(n=min_samples, random_state=42)

    all_df = pd.concat([pos_df, neg_df], ignore_index=True)
    all_df['label'] = all_df['hyperedge_label']
    print(f"Balanced: {len(all_df)} total samples")

    # Build drug index
    all_drugs = set()
    for drugs_str in all_df['DrugBankID']:
        drugs = ast.literal_eval(drugs_str)
        all_drugs.update(drugs)
    drug2idx = {d: i for i, d in enumerate(sorted(all_drugs))}
    print(f"Unique drugs: {len(drug2idx)}")

    # Build SE index and embeddings
    all_se = all_df['SE_above_0.9'].unique()
    se2idx = {se: i for i, se in enumerate(all_se)}
    print(f"Unique SEs: {len(se2idx)}")

    # Load SE name mapping
    se_dict_path = 'dataset/HODDI_v15/dictionary/Side_effects_unique.csv'
    se_dict = pd.read_csv(se_dict_path, usecols=['umls_cui_from_meddra', 'side_effect_name'])
    se_to_name = dict(zip(se_dict['umls_cui_from_meddra'], se_dict['side_effect_name']))

    print("Computing SE embeddings...")
    se_embeddings = {}
    for se in tqdm(all_se):
        se_name = se_to_name.get(se, se)
        se_embeddings[se] = get_sapbert_embedding(se_name)

    # Prepare tensors
    max_drugs = all_df['drug_count'].max()

    def prepare_sample(row):
        drugs = ast.literal_eval(row['DrugBankID'])
        drug_indices = [drug2idx[d] for d in drugs]
        padded = drug_indices + [0] * (max_drugs - len(drug_indices))
        mask = [1] * len(drug_indices) + [0] * (max_drugs - len(drug_indices))
        return padded, mask, se_embeddings[row['SE_above_0.9']], row['label'], len(drugs), se2idx[row['SE_above_0.9']]

    print("Preparing data...")
    data = [prepare_sample(row) for _, row in tqdm(all_df.iterrows(), total=len(all_df))]

    drug_indices = torch.tensor([d[0] for d in data], dtype=torch.long)
    drug_masks = torch.tensor([d[1] for d in data], dtype=torch.float)
    se_features = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float)
    labels = torch.tensor([d[3] for d in data], dtype=torch.long)
    drug_orders = np.array([d[4] for d in data])
    se_indices = np.array([d[5] for d in data])

    # Train/val/test split
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels.numpy())
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx].numpy())

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = TensorDataset(drug_indices[train_idx], drug_masks[train_idx],
                                   se_features[train_idx], labels[train_idx])
    val_dataset = TensorDataset(drug_indices[val_idx], drug_masks[val_idx],
                                 se_features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # Model
    model = DDIModel(num_drugs=len(drug2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training
    best_val_auroc = 0
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            d_idx, d_mask, se_feat, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(d_idx, d_mask, se_feat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_probs, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                d_idx, d_mask, se_feat, y = [b.to(device) for b in batch]
                logits = model(d_idx, d_mask, se_feat)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_auroc = roc_auc_score(val_labels, val_probs)
        scheduler.step(1 - val_auroc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val AUROC={val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), '3_MLP/output/v15_full_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping")
                break

    # Load best model
    model.load_state_dict(torch.load('3_MLP/output/v15_full_best.pt', weights_only=True))
    model.eval()

    def evaluate_split(split_idx, split_name):
        split_drug_idx = drug_indices[split_idx].to(device)
        split_drug_mask = drug_masks[split_idx].to(device)
        split_se_feat = se_features[split_idx].to(device)
        split_labels = labels[split_idx].numpy()
        split_orders = drug_orders[split_idx]
        split_se_idx = se_indices[split_idx]

        with torch.no_grad():
            all_logits = []
            for i in range(0, len(split_drug_idx), 512):
                logits = model(split_drug_idx[i:i+512], split_drug_mask[i:i+512], split_se_feat[i:i+512])
                all_logits.append(logits.cpu())
            all_logits = torch.cat(all_logits)
            probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
            preds = torch.argmax(all_logits, dim=-1).numpy()

        overall_basic = compute_metrics(split_labels, preds, probs)
        overall_ranking = compute_ranking_metrics(split_se_idx, split_labels, probs)

        print(f"\n=== Overall {split_name} Results ===")
        for k, v in {**overall_basic, **overall_ranking}.items():
            print(f"  {k}: {v:.4f}")

        results_by_order = {}
        for order in sorted(np.unique(split_orders)):
            mask = split_orders == order
            if mask.sum() < 10:
                continue

            y = split_labels[mask]
            pred = preds[mask]
            prob = probs[mask]
            se_idx = split_se_idx[mask]

            basic = compute_metrics(y, pred, prob)
            ranking = compute_ranking_metrics(se_idx, y, prob)

            results_by_order[int(order)] = {**basic, **ranking, 'n_samples': int(mask.sum())}

            print(f"\n=== {split_name} Drug Order {order} (n={mask.sum()}) ===")
            for k, v in results_by_order[int(order)].items():
                if k != 'n_samples':
                    print(f"  {k}: {v:.4f}")

        return {'overall': {**overall_basic, **overall_ranking}, 'by_order': results_by_order}

    print("\n" + "="*50)
    print("TRAIN SET EVALUATION")
    print("="*50)
    train_results = evaluate_split(train_idx, "Train")

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    test_results = evaluate_split(test_idx, "Test")

    output = {'train': train_results, 'test': test_results}

    os.makedirs('3_MLP/output', exist_ok=True)
    with open('3_MLP/output/v15_full_metrics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to 3_MLP/output/v15_full_metrics.json")

if __name__ == '__main__':
    os.chdir('/Users/yshi/PycharmProjects/HODDI')
    main()
