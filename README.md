# HODDI: Higher-Order Drug-Drug Interaction Prediction

This project implements drug-drug interaction (DDI) side effect prediction using three neural network architectures: MLP, GNN, and HGNN.

## Project Structure

```
HODDI/
├── dataset/
│   ├── generate_hoddi_v14.py    # Generate v14 dataset (1:X negative sampling)
│   ├── generate_hoddi_v15.py    # Generate v15 dataset (distribution-based sampling)
│   ├── HODDI_v14/               # Dataset v14 (not in git)
│   └── HODDI_v15/               # Dataset v15 (not in git)
├── 1_HGNN/                      # Hypergraph Neural Network
│   ├── models/hgnn.py           # HGNN model architecture
│   ├── utils/data_utils.py      # Data loading utilities
│   ├── main.py                  # Training script
│   └── config.py                # Configuration
├── 2_GNN/                       # Graph Neural Network
│   ├── models/gnn.py            # GNN model architecture (GAT-based)
│   ├── utils/data_utils.py      # Data loading with clique expansion
│   ├── main.py                  # Training script
│   └── config.py                # Configuration
├── 3_MLP/                       # Multi-Layer Perceptron
│   ├── models/mlp.py            # MLP model architecture
│   ├── utils/data_utils.py      # Data loading utilities
│   ├── main.py                  # Training script
│   ├── train_v14_full_metrics.py # Training with full metrics evaluation
│   └── config.py                # Configuration
└── utils/                       # Shared utilities
```

## Data Download

**Positive samples** (original HODDI dataset) can be downloaded from:
- https://github.com/TIML-Group/HODDI

After downloading, place the data in `dataset/` and run the generation scripts to create negative samples.

## Dataset Generation

### v14: 1:X Negative Sampling (Random Replacement)

For each positive sample with N drugs:
- Generate N negative samples by replacing one drug at a time (uniform random replacement)
- Generate 1 negative sample by replacing the SE (uniform random replacement)
- Total: N+1 negative samples per positive sample
- Then remove all Pos ⊂ Neg conflict samples

```bash
cd /path/to/HODDI
python dataset/generate_hoddi_v14.py
```

### v15: Distribution-Based Negative Sampling

For each positive sample with N drugs:
- Generate N negative samples by replacing each drug with a drug sampled from **positive samples' drug distribution**
- Generate 1 negative sample by replacing SE with an SE sampled from **positive samples' SE distribution**
- Total: N+1 negative samples per positive sample

This maintains more balanced SE-level ratios compared to v14.

```bash
cd /path/to/HODDI
python dataset/generate_hoddi_v15.py
```

### Dataset Statistics

| Dataset | Positive | Negative | Neg/Pos Ratio | SE-level Ratio (Mean) |
|---------|----------|----------|---------------|----------------------|
| v14 | 666,386 | 2,124,864 | 3.19 | 12.83 |
| v15 | 666,386 | 2,032,554 | 3.05 | 3.01 |

## Model Architectures

### MLP (Multi-Layer Perceptron)

- **Drug embedding**: Learnable embeddings (128-dim)
- **SE embedding**: Fixed SapBERT embeddings (768-dim)
- **Aggregation**: Mean pooling for multi-drug combinations
- **Architecture**: Drug projector + SE projector + Classifier

### GNN (Graph Neural Network)

- **Drug embedding**: Random embeddings (128-dim)
- **SE embedding**: Random embeddings (128-dim)
- **Graph construction**: Clique expansion (convert N-drug hyperedge to N*(N-1)/2 drug pairs)
- **Architecture**: GAT-based encoder + MLP classifier

### HGNN (Hypergraph Neural Network)

- **Drug embedding**: Random embeddings (128-dim)
- **SE embedding**: Random embeddings (128-dim)
- **Hypergraph**: Preserves higher-order drug interactions (no clique expansion)
- **Architecture**: HypergraphConv layers with attention + MLP classifier

## Training

### Train MLP

```bash
cd /path/to/HODDI

# Using main.py (basic training)
python 3_MLP/main.py --data_dir dataset/HODDI_v14

# Using train_v14_full_metrics.py (with comprehensive evaluation by drug order)
# Edit the data_dir in the script to switch between v14 and v15
python 3_MLP/train_v14_full_metrics.py
```

### Train GNN

```bash
cd /path/to/HODDI
python 2_GNN/main.py --data_dir dataset/HODDI_v14
```

### Train HGNN

```bash
cd /path/to/HODDI
python 1_HGNN/main.py --data_dir dataset/HODDI_v14
```

## Evaluation Metrics

The models are evaluated on the following metrics:
- **Classification**: Accuracy, Precision, Recall, F1
- **Ranking**: AUROC, AUPRC
- **Per-SE metrics**: MRR, Avg AUROC/SE, Recall@K (K=10, 30, 50)

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch >= 2.0
- PyTorch Geometric
- Transformers (for SapBERT embeddings in MLP)
- scikit-learn
- pandas
- numpy
- tqdm

## Data Format

### Input Format (HGNN/MLP)
Each sample contains:
- `DrugBankID`: List of drug IDs (e.g., `['DB00001', 'DB00002']`)
- `SE_above_0.9`: Side effect ID (e.g., `C0000727`)
- `hyperedge_label`: 1 for positive, 0 for negative

### GNN Format (Clique Expansion)
The GNN uses clique expansion to convert multi-drug interactions to pairwise:
- Original: (Drug_A, Drug_B, Drug_C) → SE
- Expanded: (Drug_A, Drug_B) → SE, (Drug_A, Drug_C) → SE, (Drug_B, Drug_C) → SE

The clique expansion code is in `2_GNN/utils/data_utils.py`.

## Notes

- Dataset files are not included in git (see `.gitignore`)
- Run the generation scripts to create negative samples after downloading positive data
- Model checkpoints are saved in `{model}/output/` directories

## License

See LICENSE.txt for details.

## Citation

```latex
@misc{wang2025hoddidatasethighorderdrugdrug,
      title={HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance}, 
      author={Zhaoying Wang and Yingdan Shi and Xiang Liu and Can Chen and Jun Wen and Ren Wang},
      journal={https://arxiv.org/abs/2502.06274},
      year={2025},
}
```
