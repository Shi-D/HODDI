"""
Data utilities for MLP model training on HODDI dataset.
Uses HGNN format (hyperedge with drug list) for direct drug aggregation.
"""

import os
import ast
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split


def get_project_root() -> str:
    """Get the project root directory."""
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    mlp_dir = os.path.dirname(utils_dir)
    project_root = os.path.dirname(mlp_dir)
    return project_root


class HODDIMLPDataset:
    """
    HODDI dataset loader for MLP model.
    Uses HGNN format where each record has a list of drugs.
    """

    def __init__(
        self,
        data_dir: str = 'dataset/HODDI_v2',
        min_drugs: int = 2,
        max_drugs: int = 42,
        max_samples_per_file: int = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to HODDI_v2 dataset directory
            min_drugs: Minimum number of drugs per hyperedge
            max_drugs: Maximum number of drugs per hyperedge
            max_samples_per_file: Maximum samples per file (for efficiency)
        """
        if not os.path.isabs(data_dir):
            project_root = get_project_root()
            data_dir = os.path.join(project_root, data_dir)

        self.data_dir = data_dir
        self.hgnn_dir = os.path.join(data_dir, 'HODDI', 'HGNN')
        self.dict_dir = os.path.join(data_dir, 'dictionary')
        self.min_drugs = min_drugs
        self.max_drugs = max_drugs
        self.max_samples_per_file = max_samples_per_file

        # Load data
        self.data = self._load_all_data()

        # Build indices
        self.drug2idx, self.idx2drug = self._build_drug_index()
        self.se2idx, self.idx2se = self._build_se_index()

        # Load SMILES
        self.smiles_dict = self._load_smiles()

    def _load_all_data(self) -> pd.DataFrame:
        """Load all data from positive and negative samples."""
        all_data = []

        pos_dir = os.path.join(self.hgnn_dir, 'positive_samples')
        neg_dir = os.path.join(self.hgnn_dir, 'negative_samples')

        for drug_count in range(self.min_drugs, self.max_drugs + 1):
            filename = f"{drug_count}_drug.csv"

            # Load positive samples
            pos_file = os.path.join(pos_dir, filename)
            if os.path.exists(pos_file):
                df = pd.read_csv(pos_file)
                if self.max_samples_per_file and len(df) > self.max_samples_per_file:
                    df = df.sample(n=self.max_samples_per_file, random_state=42)
                df['label'] = 1
                df['drug_count'] = drug_count
                all_data.append(df)

            # Load negative samples
            neg_file = os.path.join(neg_dir, filename)
            if os.path.exists(neg_file):
                df = pd.read_csv(neg_file)
                if self.max_samples_per_file and len(df) > self.max_samples_per_file:
                    df = df.sample(n=self.max_samples_per_file, random_state=42)
                df['label'] = 0
                df['drug_count'] = drug_count
                all_data.append(df)

        if not all_data:
            raise ValueError(f"No data found in {self.hgnn_dir}")

        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined)} samples from {self.min_drugs}-{self.max_drugs} drug files")
        print(f"  Positive samples: {(combined['label'] == 1).sum()}")
        print(f"  Negative samples: {(combined['label'] == 0).sum()}")

        return combined

    def _parse_drug_list(self, drug_str) -> List[str]:
        """Parse drug list from string."""
        if isinstance(drug_str, str):
            try:
                return ast.literal_eval(drug_str)
            except:
                return drug_str.strip("[]").replace("'", "").split(", ")
        return list(drug_str)

    def _build_drug_index(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build drug to index mapping."""
        drugs = set()
        for drug_list in self.data['DrugBankID']:
            parsed = self._parse_drug_list(drug_list)
            drugs.update(parsed)

        drugs = sorted(drugs)
        drug2idx = {drug: idx for idx, drug in enumerate(drugs)}
        idx2drug = {idx: drug for drug, idx in drug2idx.items()}

        print(f"Number of unique drugs: {len(drugs)}")
        return drug2idx, idx2drug

    def _build_se_index(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build side effect to index mapping."""
        # Use SE_above_0.9 column
        se_col = 'SE_above_0.9' if 'SE_above_0.9' in self.data.columns else 'SE_label'
        se_list = sorted(self.data[se_col].unique())

        se2idx = {se: idx for idx, se in enumerate(se_list)}
        idx2se = {idx: se for se, idx in se2idx.items()}

        print(f"Number of unique side effects: {len(se_list)}")
        return se2idx, idx2se

    def _load_smiles(self) -> Dict[str, str]:
        """Load drug SMILES from dictionary."""
        smiles_file = os.path.join(self.dict_dir, 'DrugBankID2SMILES.csv')
        if os.path.exists(smiles_file):
            df = pd.read_csv(smiles_file)
            id_col = 'drugbank_id' if 'drugbank_id' in df.columns else 'DrugBankID'
            smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
            return dict(zip(df[id_col], df[smiles_col]))
        return {}

    def get_data_by_drug_count(self, min_drugs: int, max_drugs: int) -> pd.DataFrame:
        """Get data filtered by drug count range."""
        mask = (self.data['drug_count'] >= min_drugs) & (self.data['drug_count'] <= max_drugs)
        return self.data[mask].copy()

    def prepare_mlp_data(
        self,
        df: pd.DataFrame,
        drug_embeddings: torch.Tensor = None,
        aggregation: str = 'mean',
        device: str = 'cpu'
    ) -> Dict:
        """
        Prepare data for MLP training.
        Pre-computes aggregated drug features for efficiency.

        Returns:
            Dict containing:
            - drug_features: Pre-aggregated drug features tensor (if embeddings provided)
            - drug_lists: List of drug index lists (fallback)
            - se_indices: Tensor of side effect indices
            - labels: Tensor of binary labels
        """
        se_col = 'SE_above_0.9' if 'SE_above_0.9' in df.columns else 'SE_label'

        drug_lists = []
        for drug_str in df['DrugBankID']:
            parsed = self._parse_drug_list(drug_str)
            indices = [self.drug2idx[d] for d in parsed if d in self.drug2idx]
            drug_lists.append(indices)

        se_indices = torch.tensor([self.se2idx[se] for se in df[se_col]], dtype=torch.long)
        labels = torch.tensor(df['label'].values, dtype=torch.long)

        result = {
            'drug_lists': drug_lists,
            'se_indices': se_indices.to(device),
            'labels': labels.to(device)
        }

        # Pre-compute aggregated drug features if embeddings provided
        if drug_embeddings is not None:
            num_samples = len(drug_lists)
            emb_dim = drug_embeddings.shape[1]
            drug_features = torch.zeros(num_samples, emb_dim)

            for i, indices in enumerate(drug_lists):
                if len(indices) > 0:
                    idx_tensor = torch.tensor(indices, dtype=torch.long)
                    embs = drug_embeddings[idx_tensor]
                    if aggregation == 'mean':
                        drug_features[i] = embs.mean(dim=0)
                    elif aggregation == 'sum':
                        drug_features[i] = embs.sum(dim=0)
                    elif aggregation == 'max':
                        drug_features[i] = embs.max(dim=0)[0]

            result['drug_features'] = drug_features.to(device)

        return result


class DataProcessor:
    """Process drug features for MLP."""

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def get_drug_embeddings(
        self,
        smiles_list: List[str],
        method: str = 'random',
        emb_dim: int = 128
    ) -> torch.Tensor:
        """Get drug embeddings."""
        num_drugs = len(smiles_list)

        if method == 'random':
            embeddings = torch.randn(num_drugs, emb_dim)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        elif method == 'morgan':
            embeddings = self._get_morgan_fingerprints(smiles_list, emb_dim)
        elif method == 'chemberta':
            embeddings = self._get_chemberta_embeddings(smiles_list, emb_dim)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        return embeddings.to(self.device)

    def _get_morgan_fingerprints(self, smiles_list: List[str], emb_dim: int) -> torch.Tensor:
        """Get Morgan fingerprints for drugs."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            print("Warning: rdkit not installed. Using random embeddings.")
            return torch.randn(len(smiles_list), emb_dim)

        fps = []
        for smiles in smiles_list:
            if smiles and pd.notna(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=emb_dim)
                    fps.append(list(fp))
                else:
                    fps.append([0] * emb_dim)
            else:
                fps.append([0] * emb_dim)

        return torch.tensor(fps, dtype=torch.float32)

    def _get_chemberta_embeddings(self, smiles_list: List[str], emb_dim: int) -> torch.Tensor:
        """Get ChemBERTa embeddings for drugs."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            print("Warning: transformers not installed. Using random embeddings.")
            return torch.randn(len(smiles_list), emb_dim)

        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for smiles in smiles_list:
                if smiles and pd.notna(smiles):
                    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze()
                    embeddings.append(emb)
                else:
                    embeddings.append(torch.zeros(768))

        embeddings = torch.stack(embeddings)

        if embeddings.shape[1] != emb_dim:
            proj = torch.nn.Linear(embeddings.shape[1], emb_dim)
            embeddings = proj(embeddings)

        return embeddings


def prepare_data(
    data_dir: str = 'dataset/HODDI_v2',
    min_drugs: int = 2,
    max_drugs: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    balance: bool = True,
    device: str = 'cpu',
    random_state: int = 42,
    max_samples_per_file: int = None,
    drug_emb_method: str = 'random',
    drug_emb_dim: int = 128,
    aggregation: str = 'mean'
) -> Dict:
    """Prepare data with random split."""
    dataset = HODDIMLPDataset(data_dir, min_drugs, max_drugs, max_samples_per_file)

    data = dataset.data.copy()

    if balance:
        pos_data = data[data['label'] == 1]
        neg_data = data[data['label'] == 0]
        min_samples = min(len(pos_data), len(neg_data))
        pos_data = pos_data.sample(n=min_samples, random_state=random_state)
        neg_data = neg_data.sample(n=min_samples, random_state=random_state)
        data = pd.concat([pos_data, neg_data], ignore_index=True)
        print(f"Balanced data: {len(data)} samples ({min_samples} pos, {min_samples} neg)")

    train_data, temp_data = train_test_split(
        data, train_size=train_ratio, random_state=random_state, stratify=data['label']
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, train_size=val_size, random_state=random_state, stratify=temp_data['label']
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Compute drug embeddings
    print(f"\nComputing drug embeddings (method: {drug_emb_method})...")
    processor = DataProcessor(device)
    smiles_list = [dataset.smiles_dict.get(drug, '') for drug in dataset.idx2drug.values()]
    drug_embeddings = processor.get_drug_embeddings(smiles_list, drug_emb_method, drug_emb_dim)

    print(f"Pre-computing aggregated drug features (aggregation: {aggregation})...")

    return {
        'train': dataset.prepare_mlp_data(train_data, drug_embeddings, aggregation, device),
        'val': dataset.prepare_mlp_data(val_data, drug_embeddings, aggregation, device),
        'test': dataset.prepare_mlp_data(test_data, drug_embeddings, aggregation, device),
        'num_drugs': len(dataset.drug2idx),
        'num_se': len(dataset.se2idx),
        'drug2idx': dataset.drug2idx,
        'se2idx': dataset.se2idx,
        'smiles_list': smiles_list,
        'processor': processor
    }


def prepare_data_by_drug_count(
    data_dir: str = 'dataset/HODDI_v2',
    train_val_drug_range: Tuple[int, int] = (2, 5),
    test_drug_range: Tuple[int, int] = (6, 42),
    train_val_ratio: Tuple[float, float] = (0.7, 0.2),
    balance: bool = True,
    device: str = 'cpu',
    random_state: int = 42,
    max_samples_per_file: int = None,
    drug_emb_method: str = 'random',
    drug_emb_dim: int = 128,
    aggregation: str = 'mean'
) -> Dict:
    """Prepare data split by drug count."""
    min_drugs = min(train_val_drug_range[0], test_drug_range[0])
    max_drugs = max(train_val_drug_range[1], test_drug_range[1])

    dataset = HODDIMLPDataset(data_dir, min_drugs, max_drugs, max_samples_per_file)

    train_val_data = dataset.get_data_by_drug_count(train_val_drug_range[0], train_val_drug_range[1])
    test_data = dataset.get_data_by_drug_count(test_drug_range[0], test_drug_range[1])

    print(f"\nData by drug count:")
    print(f"  Train/Val ({train_val_drug_range[0]}-{train_val_drug_range[1]} drugs): {len(train_val_data)} samples")
    print(f"  Test ({test_drug_range[0]}-{test_drug_range[1]} drugs): {len(test_data)} samples")

    if balance:
        pos_data = train_val_data[train_val_data['label'] == 1]
        neg_data = train_val_data[train_val_data['label'] == 0]
        min_samples = min(len(pos_data), len(neg_data))
        pos_data = pos_data.sample(n=min_samples, random_state=random_state)
        neg_data = neg_data.sample(n=min_samples, random_state=random_state)
        train_val_data = pd.concat([pos_data, neg_data], ignore_index=True)
        print(f"  Balanced train/val: {len(train_val_data)} samples")

    train_ratio = train_val_ratio[0] / (train_val_ratio[0] + train_val_ratio[1])
    train_data, val_data = train_test_split(
        train_val_data, train_size=train_ratio, random_state=random_state,
        stratify=train_val_data['label']
    )

    if balance:
        pos_test = test_data[test_data['label'] == 1]
        neg_test = test_data[test_data['label'] == 0]
        min_samples = min(len(pos_test), len(neg_test))
        if min_samples > 0:
            pos_test = pos_test.sample(n=min_samples, random_state=random_state)
            neg_test = neg_test.sample(n=min_samples, random_state=random_state)
            test_data = pd.concat([pos_test, neg_test], ignore_index=True)
            print(f"  Balanced test: {len(test_data)} samples")

    print(f"\nFinal split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Compute drug embeddings
    print(f"\nComputing drug embeddings (method: {drug_emb_method})...")
    processor = DataProcessor(device)
    smiles_list = [dataset.smiles_dict.get(drug, '') for drug in dataset.idx2drug.values()]
    drug_embeddings = processor.get_drug_embeddings(smiles_list, drug_emb_method, drug_emb_dim)

    print(f"Pre-computing aggregated drug features (aggregation: {aggregation})...")

    return {
        'train': dataset.prepare_mlp_data(train_data, drug_embeddings, aggregation, device),
        'val': dataset.prepare_mlp_data(val_data, drug_embeddings, aggregation, device),
        'test': dataset.prepare_mlp_data(test_data, drug_embeddings, aggregation, device),
        'num_drugs': len(dataset.drug2idx),
        'num_se': len(dataset.se2idx),
        'drug2idx': dataset.drug2idx,
        'se2idx': dataset.se2idx,
        'smiles_list': smiles_list,
        'processor': processor
    }
