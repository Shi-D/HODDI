"""
Data utilities for HGNN model.
Handles data loading, preprocessing, and train/val/test splitting.
"""

import os
import ast
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_project_root() -> str:
    """Get the project root directory."""
    # This file is at 1_HGNN/utils/data_utils.py
    # Project root is two levels up
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    hgnn_dir = os.path.dirname(utils_dir)
    project_root = os.path.dirname(hgnn_dir)
    return project_root


class HODDIDataset:
    """Dataset class for HODDI data loading and preprocessing."""

    def __init__(
        self,
        data_dir: str = "dataset/HODDI_v2",
        min_drugs: int = 2,
        max_drugs: int = 42,
        use_gnn_format: bool = False
    ):
        """
        Initialize the HODDI dataset.

        Args:
            data_dir: Path to HODDI_v2 dataset directory (relative to project root or absolute)
            min_drugs: Minimum number of drugs per hyperedge
            max_drugs: Maximum number of drugs per hyperedge
            use_gnn_format: If True, use GNN format; otherwise use HGNN format
        """
        # Handle relative paths - make them relative to project root
        if not os.path.isabs(data_dir):
            project_root = get_project_root()
            data_dir = os.path.join(project_root, data_dir)

        self.data_dir = data_dir
        self.min_drugs = min_drugs
        self.max_drugs = max_drugs
        self.format_dir = "GNN" if use_gnn_format else "HGNN"

        # Verify data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Will be populated during loading
        self.drug_to_idx: Dict[str, int] = {}
        self.idx_to_drug: Dict[int, str] = {}
        self.se_to_idx: Dict[str, int] = {}
        self.idx_to_se: Dict[int, str] = {}
        self.drug_smiles: Dict[str, str] = {}

        # Load dictionaries
        self._load_dictionaries()

    def _load_dictionaries(self):
        """Load drug SMILES and side effect mappings."""
        # Load drug SMILES
        smiles_path = os.path.join(self.data_dir, "dictionary", "DrugBankID2SMILES.csv")
        if os.path.exists(smiles_path):
            smiles_df = pd.read_csv(smiles_path)
            for _, row in smiles_df.iterrows():
                drug_id = row['drugbank_id']
                smiles = row['smiles']
                if pd.notna(smiles) and smiles.strip():
                    self.drug_smiles[drug_id] = smiles

        # Load unique side effects
        se_path = os.path.join(self.data_dir, "dictionary", "SE_above_0.9_unique_values.csv")
        if os.path.exists(se_path):
            se_df = pd.read_csv(se_path)
            col_name = se_df.columns[0]
            for idx, se in enumerate(se_df[col_name].values):
                self.se_to_idx[se] = idx
                self.idx_to_se[idx] = se

        print(f"Loaded {len(self.drug_smiles)} drugs with SMILES")
        print(f"Loaded {len(self.se_to_idx)} unique side effects")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load positive and negative samples from all drug files.

        Returns:
            Tuple of (positive_df, negative_df)
        """
        pos_samples = []
        neg_samples = []

        pos_dir = os.path.join(self.data_dir, "HODDI", self.format_dir, "positive_samples")
        neg_dir = os.path.join(self.data_dir, "HODDI", self.format_dir, "negative_samples")

        print(f"Looking for data in:")
        print(f"  Positive: {pos_dir}")
        print(f"  Negative: {neg_dir}")

        # Check if directories exist
        if not os.path.exists(pos_dir):
            raise FileNotFoundError(f"Positive samples directory not found: {pos_dir}")
        if not os.path.exists(neg_dir):
            raise FileNotFoundError(f"Negative samples directory not found: {neg_dir}")

        files_found = 0
        for n_drugs in tqdm(range(self.min_drugs, self.max_drugs + 1), desc="Loading data"):
            pos_file = os.path.join(pos_dir, f"{n_drugs}_drug.csv")
            neg_file = os.path.join(neg_dir, f"{n_drugs}_drug.csv")

            if os.path.exists(pos_file):
                df = pd.read_csv(pos_file)
                df['n_drugs'] = n_drugs
                pos_samples.append(df)
                files_found += 1

            if os.path.exists(neg_file):
                df = pd.read_csv(neg_file)
                df['n_drugs'] = n_drugs
                neg_samples.append(df)
                files_found += 1

        if files_found == 0:
            raise FileNotFoundError(
                f"No data files found! Check that files like '2_drug.csv' exist in {pos_dir}"
            )

        pos_df = pd.concat(pos_samples, ignore_index=True) if pos_samples else pd.DataFrame()
        neg_df = pd.concat(neg_samples, ignore_index=True) if neg_samples else pd.DataFrame()

        print(f"Loaded {len(pos_df)} positive samples")
        print(f"Loaded {len(neg_df)} negative samples")

        return pos_df, neg_df

    def build_drug_index(self, pos_df: pd.DataFrame, neg_df: pd.DataFrame):
        """Build drug to index mapping from the data."""
        all_drugs = set()

        for df in [pos_df, neg_df]:
            if 'DrugBankID' in df.columns:
                for drug_list in df['DrugBankID']:
                    drugs = ast.literal_eval(drug_list) if isinstance(drug_list, str) else drug_list
                    all_drugs.update(drugs)

        # Filter to drugs with SMILES
        drugs_with_smiles = [d for d in all_drugs if d in self.drug_smiles]

        for idx, drug in enumerate(sorted(drugs_with_smiles)):
            self.drug_to_idx[drug] = idx
            self.idx_to_drug[idx] = drug

        print(f"Built index for {len(self.drug_to_idx)} drugs (with SMILES)")

    def prepare_samples(
        self,
        pos_df: pd.DataFrame,
        neg_df: pd.DataFrame,
        balance: bool = True
    ) -> pd.DataFrame:
        """
        Prepare and optionally balance positive and negative samples.

        Args:
            pos_df: Positive samples DataFrame
            neg_df: Negative samples DataFrame
            balance: If True, balance positive and negative samples

        Returns:
            Combined DataFrame with labels
        """
        pos_df = pos_df.copy()
        neg_df = neg_df.copy()

        pos_df['label'] = 1
        neg_df['label'] = 0

        if balance:
            min_samples = min(len(pos_df), len(neg_df))
            pos_df = pos_df.sample(n=min_samples, random_state=42)
            neg_df = neg_df.sample(n=min_samples, random_state=42)
            print(f"Balanced to {min_samples} samples each")

        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return combined_df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: DataFrame to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1"

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=df['label']
        )

        # Second split: val vs test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_ratio,
            random_state=random_state,
            stratify=temp_df['label']
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def build_incidence_matrix(
        self,
        df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build incidence matrix for hypergraph.

        The incidence matrix H has shape (num_drugs, num_hyperedges),
        where H[i, j] = 1 if drug i is in hyperedge j.

        Args:
            df: DataFrame with 'DrugBankID' column

        Returns:
            Tuple of (incidence_matrix, se_indices, labels)
        """
        num_drugs = len(self.drug_to_idx)
        num_edges = len(df)

        # Initialize incidence matrix
        incidence = torch.zeros(num_drugs, num_edges, dtype=torch.float32)
        se_indices = torch.zeros(num_edges, dtype=torch.long)
        labels = torch.zeros(num_edges, dtype=torch.long)

        valid_edges = []

        for edge_idx, (_, row) in enumerate(df.iterrows()):
            # Parse drug list
            drug_list = ast.literal_eval(row['DrugBankID']) if isinstance(row['DrugBankID'], str) else row['DrugBankID']

            # Filter to drugs in our index
            valid_drugs = [d for d in drug_list if d in self.drug_to_idx]

            if len(valid_drugs) < 2:
                continue

            valid_edges.append(edge_idx)

            # Fill incidence matrix
            for drug in valid_drugs:
                drug_idx = self.drug_to_idx[drug]
                incidence[drug_idx, edge_idx] = 1.0

            # Side effect index
            se = row['SE_above_0.9']
            se_indices[edge_idx] = self.se_to_idx.get(se, 0)

            # Label
            labels[edge_idx] = row['label']

        # Filter to valid edges
        valid_edges = torch.tensor(valid_edges, dtype=torch.long)
        incidence = incidence[:, valid_edges]
        se_indices = se_indices[valid_edges]
        labels = labels[valid_edges]

        print(f"Built incidence matrix: {incidence.shape}")

        return incidence, se_indices, labels

    def get_drug_smiles_list(self) -> List[str]:
        """Get ordered list of SMILES strings for all indexed drugs."""
        smiles_list = []
        for idx in range(len(self.drug_to_idx)):
            drug = self.idx_to_drug[idx]
            smiles_list.append(self.drug_smiles.get(drug, ""))
        return smiles_list


class DataProcessor:
    """Process data for HGNN model."""

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def get_drug_embeddings(
        self,
        smiles_list: List[str],
        method: str = 'random',
        emb_dim: int = 128
    ) -> torch.Tensor:
        """
        Get drug embeddings from SMILES.

        Args:
            smiles_list: List of SMILES strings
            method: Embedding method ('random', 'morgan', 'chemberta')
            emb_dim: Embedding dimension

        Returns:
            Tensor of shape (num_drugs, emb_dim)
        """
        num_drugs = len(smiles_list)

        if method == 'random':
            # Random embeddings for testing
            embeddings = torch.randn(num_drugs, emb_dim)

        elif method == 'morgan':
            # Morgan fingerprints
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                embeddings = torch.zeros(num_drugs, emb_dim)
                for i, smiles in enumerate(tqdm(smiles_list, desc="Computing Morgan FP")):
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=emb_dim)
                            embeddings[i] = torch.tensor(list(fp), dtype=torch.float32)
            except ImportError:
                print("RDKit not installed, using random embeddings")
                embeddings = torch.randn(num_drugs, emb_dim)

        elif method == 'chemberta':
            # ChemBERTa embeddings
            try:
                from transformers import AutoTokenizer, AutoModel

                tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                model.eval()

                embeddings = []
                with torch.no_grad():
                    for smiles in tqdm(smiles_list, desc="Computing ChemBERTa embeddings"):
                        if smiles:
                            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)
                            outputs = model(**inputs)
                            emb = outputs.last_hidden_state.mean(dim=1).squeeze()
                        else:
                            emb = torch.zeros(768)
                        embeddings.append(emb)

                embeddings = torch.stack(embeddings)
                # Project to target dimension
                proj = torch.nn.Linear(768, emb_dim)
                embeddings = proj(embeddings)

            except ImportError:
                print("Transformers not installed, using random embeddings")
                embeddings = torch.randn(num_drugs, emb_dim)

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        return embeddings.to(self.device)

    def get_se_embeddings(
        self,
        num_se: int,
        method: str = 'onehot',
        emb_dim: int = 128
    ) -> torch.Tensor:
        """
        Get side effect embeddings.

        Args:
            num_se: Number of unique side effects
            method: Embedding method ('onehot', 'random', 'learnable')
            emb_dim: Embedding dimension

        Returns:
            Tensor of shape (num_se, emb_dim) or nn.Embedding
        """
        if method == 'onehot':
            # One-hot encoding projected to emb_dim
            onehot = torch.eye(num_se)
            proj = torch.nn.Linear(num_se, emb_dim)
            embeddings = proj(onehot)

        elif method == 'random':
            embeddings = torch.randn(num_se, emb_dim)

        elif method == 'learnable':
            # Return nn.Embedding for learnable embeddings
            embeddings = torch.nn.Embedding(num_se, emb_dim)
            return embeddings.to(self.device)

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        return embeddings.to(self.device)


def collate_fn(batch):
    """Collate function for DataLoader."""
    incidence_matrices = [item['incidence'] for item in batch]
    se_indices = [item['se_idx'] for item in batch]
    labels = [item['label'] for item in batch]

    return {
        'incidence': torch.stack(incidence_matrices),
        'se_idx': torch.stack(se_indices),
        'label': torch.stack(labels)
    }


def prepare_data(
    data_dir: str = "dataset/HODDI_v2",
    min_drugs: int = 2,
    max_drugs: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    balance: bool = True,
    device: str = 'cpu'
) -> dict:
    """
    Prepare all data for training.

    Returns:
        Dictionary containing all processed data
    """
    # Initialize dataset
    dataset = HODDIDataset(data_dir, min_drugs, max_drugs)

    # Load data
    pos_df, neg_df = dataset.load_data()

    # Build drug index
    dataset.build_drug_index(pos_df, neg_df)

    # Prepare samples
    combined_df = dataset.prepare_samples(pos_df, neg_df, balance=balance)

    # Split data
    train_df, val_df, test_df = dataset.split_data(
        combined_df, train_ratio, val_ratio, test_ratio
    )

    # Build incidence matrices
    train_incidence, train_se, train_labels = dataset.build_incidence_matrix(train_df)
    val_incidence, val_se, val_labels = dataset.build_incidence_matrix(val_df)
    test_incidence, test_se, test_labels = dataset.build_incidence_matrix(test_df)

    # Get embeddings
    processor = DataProcessor(device)
    smiles_list = dataset.get_drug_smiles_list()

    return {
        'dataset': dataset,
        'train': {
            'incidence': train_incidence.to(device),
            'se_indices': train_se.to(device),
            'labels': train_labels.to(device)
        },
        'val': {
            'incidence': val_incidence.to(device),
            'se_indices': val_se.to(device),
            'labels': val_labels.to(device)
        },
        'test': {
            'incidence': test_incidence.to(device),
            'se_indices': test_se.to(device),
            'labels': test_labels.to(device)
        },
        'num_drugs': len(dataset.drug_to_idx),
        'num_se': len(dataset.se_to_idx),
        'smiles_list': smiles_list,
        'processor': processor
    }


def prepare_data_by_drug_count(
    data_dir: str = "dataset/HODDI_v2",
    train_val_drug_range: Tuple[int, int] = (2, 5),
    test_drug_range: Tuple[int, int] = (6, 42),
    train_val_ratio: Tuple[float, float] = (0.7, 0.2),
    balance: bool = True,
    device: str = 'cpu',
    random_state: int = 42
) -> dict:
    """
    Prepare data split by drug count.

    Train and Val data come from hyperedges with drug count in train_val_drug_range,
    Test data comes from hyperedges with drug count in test_drug_range.

    Args:
        data_dir: Path to HODDI_v2 dataset directory
        train_val_drug_range: (min, max) drug count for train/val data (inclusive)
        test_drug_range: (min, max) drug count for test data (inclusive)
        train_val_ratio: (train_ratio, val_ratio) within train_val data, should sum to < 1
        balance: Whether to balance positive and negative samples
        device: Device to use
        random_state: Random seed

    Returns:
        Dictionary containing all processed data
    """
    train_val_min, train_val_max = train_val_drug_range
    test_min, test_max = test_drug_range
    train_ratio, val_ratio = train_val_ratio

    print("=" * 60)
    print("Data Split by Drug Count")
    print("=" * 60)
    print(f"Train/Val: {train_val_min}-{train_val_max} drugs (ratio {train_ratio}:{val_ratio})")
    print(f"Test: {test_min}-{test_max} drugs")

    # Initialize dataset with full range to load all necessary dictionaries
    full_min = min(train_val_min, test_min)
    full_max = max(train_val_max, test_max)
    dataset = HODDIDataset(data_dir, full_min, full_max)

    # Load train/val data (2-5 drugs)
    print(f"\nLoading train/val data ({train_val_min}-{train_val_max} drugs)...")
    train_val_pos = []
    train_val_neg = []

    pos_dir = os.path.join(dataset.data_dir, "HODDI", dataset.format_dir, "positive_samples")
    neg_dir = os.path.join(dataset.data_dir, "HODDI", dataset.format_dir, "negative_samples")

    for n_drugs in tqdm(range(train_val_min, train_val_max + 1), desc="Loading train/val"):
        pos_file = os.path.join(pos_dir, f"{n_drugs}_drug.csv")
        neg_file = os.path.join(neg_dir, f"{n_drugs}_drug.csv")

        if os.path.exists(pos_file):
            df = pd.read_csv(pos_file)
            df['n_drugs'] = n_drugs
            train_val_pos.append(df)

        if os.path.exists(neg_file):
            df = pd.read_csv(neg_file)
            df['n_drugs'] = n_drugs
            train_val_neg.append(df)

    train_val_pos_df = pd.concat(train_val_pos, ignore_index=True) if train_val_pos else pd.DataFrame()
    train_val_neg_df = pd.concat(train_val_neg, ignore_index=True) if train_val_neg else pd.DataFrame()

    print(f"  Positive: {len(train_val_pos_df)}, Negative: {len(train_val_neg_df)}")

    # Load test data (6-42 drugs)
    print(f"\nLoading test data ({test_min}-{test_max} drugs)...")
    test_pos = []
    test_neg = []

    for n_drugs in tqdm(range(test_min, test_max + 1), desc="Loading test"):
        pos_file = os.path.join(pos_dir, f"{n_drugs}_drug.csv")
        neg_file = os.path.join(neg_dir, f"{n_drugs}_drug.csv")

        if os.path.exists(pos_file):
            df = pd.read_csv(pos_file)
            df['n_drugs'] = n_drugs
            test_pos.append(df)

        if os.path.exists(neg_file):
            df = pd.read_csv(neg_file)
            df['n_drugs'] = n_drugs
            test_neg.append(df)

    test_pos_df = pd.concat(test_pos, ignore_index=True) if test_pos else pd.DataFrame()
    test_neg_df = pd.concat(test_neg, ignore_index=True) if test_neg else pd.DataFrame()

    print(f"  Positive: {len(test_pos_df)}, Negative: {len(test_neg_df)}")

    # Build drug index from all data
    all_pos = pd.concat([train_val_pos_df, test_pos_df], ignore_index=True)
    all_neg = pd.concat([train_val_neg_df, test_neg_df], ignore_index=True)
    dataset.build_drug_index(all_pos, all_neg)

    # Prepare train/val samples
    train_val_pos_df = train_val_pos_df.copy()
    train_val_neg_df = train_val_neg_df.copy()
    train_val_pos_df['label'] = 1
    train_val_neg_df['label'] = 0

    if balance:
        min_samples = min(len(train_val_pos_df), len(train_val_neg_df))
        train_val_pos_df = train_val_pos_df.sample(n=min_samples, random_state=random_state)
        train_val_neg_df = train_val_neg_df.sample(n=min_samples, random_state=random_state)
        print(f"\nBalanced train/val to {min_samples} samples each")

    train_val_df = pd.concat([train_val_pos_df, train_val_neg_df], ignore_index=True)
    train_val_df = train_val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split train/val (7:2 ratio -> train=7/9, val=2/9)
    total_ratio = train_ratio + val_ratio
    actual_train_ratio = train_ratio / total_ratio

    train_df, val_df = train_test_split(
        train_val_df,
        train_size=actual_train_ratio,
        random_state=random_state,
        stratify=train_val_df['label']
    )

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")

    # Prepare test samples
    test_pos_df = test_pos_df.copy()
    test_neg_df = test_neg_df.copy()
    test_pos_df['label'] = 1
    test_neg_df['label'] = 0

    if balance:
        min_samples = min(len(test_pos_df), len(test_neg_df))
        test_pos_df = test_pos_df.sample(n=min_samples, random_state=random_state)
        test_neg_df = test_neg_df.sample(n=min_samples, random_state=random_state)
        print(f"Balanced test to {min_samples} samples each")

    test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Test: {len(test_df)}")

    # Build incidence matrices
    print("\nBuilding incidence matrices...")
    train_incidence, train_se, train_labels = dataset.build_incidence_matrix(train_df)
    val_incidence, val_se, val_labels = dataset.build_incidence_matrix(val_df)
    test_incidence, test_se, test_labels = dataset.build_incidence_matrix(test_df)

    # Get embeddings
    processor = DataProcessor(device)
    smiles_list = dataset.get_drug_smiles_list()

    print("\n" + "=" * 60)
    print("Data Summary:")
    print(f"  Train: {train_incidence.shape[1]} samples")
    print(f"  Val: {val_incidence.shape[1]} samples")
    print(f"  Test: {test_incidence.shape[1]} samples")
    print(f"  Drugs: {len(dataset.drug_to_idx)}")
    print(f"  Side Effects: {len(dataset.se_to_idx)}")
    print("=" * 60)

    return {
        'dataset': dataset,
        'train': {
            'incidence': train_incidence.to(device),
            'se_indices': train_se.to(device),
            'labels': train_labels.to(device)
        },
        'val': {
            'incidence': val_incidence.to(device),
            'se_indices': val_se.to(device),
            'labels': val_labels.to(device)
        },
        'test': {
            'incidence': test_incidence.to(device),
            'se_indices': test_se.to(device),
            'labels': test_labels.to(device)
        },
        'num_drugs': len(dataset.drug_to_idx),
        'num_se': len(dataset.se_to_idx),
        'smiles_list': smiles_list,
        'processor': processor,
        'split_info': {
            'train_val_drug_range': train_val_drug_range,
            'test_drug_range': test_drug_range,
            'train_val_ratio': train_val_ratio
        }
    }
