"""
Fast Drug Interaction Transformer (DIT-Fast) for DDI side effect prediction.

Optimized version that uses pure matrix operations, no Python loops.
Key insight: Use incidence matrix directly for weighted aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class WeightedDrugAggregator(nn.Module):
    """
    Aggregate drug embeddings using incidence matrix weights.
    Pure matrix operations, no loops.
    """

    def __init__(self, drug_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, drug_dim) - all drug embeddings
            incidence: (num_drugs, batch_size) - incidence matrix
        Returns:
            (batch_size, hidden_dim) - aggregated drug features
        """
        # Project drug embeddings
        drug_feat = self.drug_proj(drug_emb)  # (num_drugs, hidden_dim)

        # Weighted aggregation: incidence.T @ drug_feat
        # (batch_size, num_drugs) @ (num_drugs, hidden_dim) = (batch_size, hidden_dim)
        aggregated = torch.mm(incidence.t(), drug_feat)

        # Normalize by number of drugs per sample
        num_drugs = incidence.sum(dim=0, keepdim=True).t().clamp(min=1)  # (batch_size, 1)
        aggregated = aggregated / num_drugs

        aggregated = self.norm(aggregated)
        aggregated = self.dropout(aggregated)

        return aggregated


class DrugInteractionLayer(nn.Module):
    """
    Learn drug interactions through learned attention over drug embeddings.
    Uses SE embedding to modulate attention.
    """

    def __init__(self, drug_dim: int, se_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query from SE, Key/Value from drugs
        self.se_to_query = nn.Linear(se_dim, hidden_dim)
        self.drug_to_key = nn.Linear(drug_dim, hidden_dim)
        self.drug_to_value = nn.Linear(drug_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        drug_emb: torch.Tensor,
        se_emb: torch.Tensor,
        incidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, drug_dim)
            se_emb: (batch_size, se_dim)
            incidence: (num_drugs, batch_size)
        Returns:
            (batch_size, hidden_dim)
        """
        batch_size = se_emb.shape[0]
        num_drugs = drug_emb.shape[0]

        # Compute keys and values for all drugs
        keys = self.drug_to_key(drug_emb)  # (num_drugs, hidden_dim)
        values = self.drug_to_value(drug_emb)  # (num_drugs, hidden_dim)

        # Compute query from SE
        query = self.se_to_query(se_emb)  # (batch_size, hidden_dim)

        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        keys = keys.view(num_drugs, self.num_heads, self.head_dim)  # (num_drugs, heads, head_dim)
        values = values.view(num_drugs, self.num_heads, self.head_dim)  # (num_drugs, heads, head_dim)

        # Compute attention scores: query @ keys.T
        # (batch, heads, head_dim) @ (head_dim, num_drugs, heads) -> need einsum
        # attn[b, h, d] = sum_k query[b, h, k] * keys[d, h, k]
        attn = torch.einsum('bhk,dhk->bhd', query, keys) / self.scale  # (batch, heads, num_drugs)

        # Mask: only attend to drugs present in this sample
        # incidence: (num_drugs, batch_size) -> need (batch, 1, num_drugs)
        mask = incidence.t().unsqueeze(1)  # (batch, 1, num_drugs)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax over drugs
        attn = F.softmax(attn, dim=-1)  # (batch, heads, num_drugs)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle case where all drugs are masked
        attn = self.dropout(attn)

        # Apply attention to values
        # out[b, h, k] = sum_d attn[b, h, d] * values[d, h, k]
        out = torch.einsum('bhd,dhk->bhk', attn, values)  # (batch, heads, head_dim)

        # Reshape and project
        out = out.reshape(batch_size, self.hidden_dim)
        out = self.out_proj(out)
        out = self.norm(out)
        out = self.dropout(out)

        return out


class PairwiseInteractionFast(nn.Module):
    """
    Fast pairwise drug interaction using outer product approximation.
    No Python loops - pure matrix operations.
    """

    def __init__(self, drug_dim: int, hidden_dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank

        # Low-rank factorization for efficiency
        self.drug_proj_a = nn.Linear(drug_dim, rank)
        self.drug_proj_b = nn.Linear(drug_dim, rank)
        self.out_proj = nn.Linear(rank, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, drug_dim)
            incidence: (num_drugs, batch_size)
        Returns:
            (batch_size, hidden_dim)
        """
        # Project drugs to two low-rank spaces
        drug_a = self.drug_proj_a(drug_emb)  # (num_drugs, rank)
        drug_b = self.drug_proj_b(drug_emb)  # (num_drugs, rank)

        # Weighted aggregation for each sample
        # sum_a[b] = sum_d incidence[d, b] * drug_a[d]
        sum_a = torch.mm(incidence.t(), drug_a)  # (batch, rank)
        sum_b = torch.mm(incidence.t(), drug_b)  # (batch, rank)

        # Pairwise interaction approximation:
        # For drugs d1, d2 in sample: drug_a[d1] * drug_b[d2]
        # sum over all pairs ≈ (sum_d drug_a[d]) * (sum_d drug_b[d]) - sum_d (drug_a[d] * drug_b[d])
        # The second term removes self-interactions

        # Self-interaction: sum_d incidence[d, b] * drug_a[d] * drug_b[d]
        self_interact = torch.mm(incidence.t(), drug_a * drug_b)  # (batch, rank)

        # Pairwise = outer - self
        pairwise = sum_a * sum_b - self_interact  # (batch, rank)

        # Normalize by number of pairs
        num_drugs = incidence.sum(dim=0).clamp(min=1)  # (batch,)
        num_pairs = (num_drugs * (num_drugs - 1) / 2).clamp(min=1).unsqueeze(1)  # (batch, 1)
        pairwise = pairwise / num_pairs

        # Project out
        out = self.out_proj(pairwise)
        out = self.norm(out)
        out = self.dropout(out)

        return out


class SEEncoder(nn.Module):
    """Encode side effect embeddings."""

    def __init__(self, num_se: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_se, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, se_indices: torch.Tensor) -> torch.Tensor:
        se_emb = self.embedding(se_indices)
        return self.proj(se_emb)


class BilinearFusionFast(nn.Module):
    """Fast bilinear fusion using low-rank factorization."""

    def __init__(self, dim1: int, dim2: int, output_dim: int, rank: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj1 = nn.Linear(dim1, rank)
        self.proj2 = nn.Linear(dim2, rank)
        self.out = nn.Linear(rank, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.proj1(x1))
        h2 = torch.tanh(self.proj2(x2))
        fused = h1 * h2
        out = self.out(fused)
        out = self.norm(out)
        return self.dropout(out)


class DITFast(nn.Module):
    """
    Fast Drug Interaction Transformer.

    All operations are pure matrix operations - no Python loops.
    Uses incidence matrix directly for efficient computation.

    Architecture:
    1. Weighted drug aggregation (mean pooling via incidence matrix)
    2. SE-guided attention over drugs
    3. Fast pairwise interaction (outer product approximation)
    4. Bilinear fusion of drug features with SE
    5. Classification
    """

    def __init__(
        self,
        num_drugs: int,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        drug_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.hidden_dim = hidden_dim

        # Drug embeddings
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_dim = drug_emb_dim

        # Drug projection
        self.drug_proj = nn.Sequential(
            nn.Linear(actual_drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # SE encoder
        self.se_encoder = SEEncoder(num_se, se_emb_dim, hidden_dim, dropout)

        # Drug aggregation (mean pooling)
        self.drug_aggregator = WeightedDrugAggregator(hidden_dim, hidden_dim, dropout)

        # SE-guided drug attention
        self.se_guided_attn = DrugInteractionLayer(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)

        # Fast pairwise interaction
        self.pairwise = PairwiseInteractionFast(hidden_dim, hidden_dim, rank=32, dropout=dropout)

        # Fusion layers
        self.drug_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Drug-SE bilinear fusion
        self.bilinear_fusion = BilinearFusionFast(hidden_dim, hidden_dim, hidden_dim, rank=64, dropout=dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        incidence_matrix: torch.Tensor,
        se_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            incidence_matrix: (num_drugs, batch_size)
            se_indices: (batch_size,)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        # Project drug embeddings
        drug_emb = self.drug_proj(drug_emb)  # (num_drugs, hidden_dim)

        # Get SE embeddings
        se_feat = self.se_encoder(se_indices)  # (batch, hidden_dim)

        # 1. Weighted drug aggregation (mean pooling)
        drug_mean = self.drug_aggregator(drug_emb, incidence_matrix)  # (batch, hidden_dim)

        # 2. SE-guided attention over drugs
        drug_attn = self.se_guided_attn(drug_emb, se_feat, incidence_matrix)  # (batch, hidden_dim)

        # 3. Fast pairwise interaction
        drug_pair = self.pairwise(drug_emb, incidence_matrix)  # (batch, hidden_dim)

        # 4. Fuse drug features
        drug_combined = torch.cat([drug_mean, drug_attn, drug_pair], dim=-1)
        drug_fused = self.drug_fusion(drug_combined)  # (batch, hidden_dim)

        # 5. Bilinear fusion with SE
        drug_se_fused = self.bilinear_fusion(drug_fused, se_feat)  # (batch, hidden_dim)

        # 6. Combine and classify
        combined = torch.cat([drug_fused, drug_se_fused], dim=-1)  # (batch, hidden_dim*2)
        logits = self.classifier(combined)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
