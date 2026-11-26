"""
HGNN v5: Cross-Modal Drug-SE Interaction Network

Key innovations:
1. SE-conditioned drug encoding: SE guides how to encode drugs
2. Cross-attention: bidirectional attention between drugs and SE
3. Drug importance scoring: learn which drugs matter for each SE
4. Gated fusion: adaptive combination of different features

Design philosophy:
- Keep it simple but targeted
- SE should actively participate in drug encoding (not just concat at end)
- Model drug importance for specific SEs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class SEConditionedDrugEncoder(nn.Module):
    """
    Encode drugs conditioned on SE.
    Different SEs focus on different aspects of drug features.

    Uses FiLM (Feature-wise Linear Modulation) conditioning.
    """

    def __init__(self, drug_dim: int, se_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Drug projection
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)

        # SE generates modulation parameters (gamma, beta) for FiLM
        self.se_to_gamma = nn.Linear(se_dim, hidden_dim)
        self.se_to_beta = nn.Linear(se_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_emb: torch.Tensor, se_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, drug_dim)
            se_emb: (batch_size, se_dim)
            incidence: (num_drugs, batch_size)
        Returns:
            (batch_size, hidden_dim) - SE-conditioned drug representation
        """
        batch_size = se_emb.shape[0]

        # Project drugs
        drug_feat = self.drug_proj(drug_emb)  # (num_drugs, hidden_dim)

        # Aggregate drugs for each sample
        # (batch, num_drugs) @ (num_drugs, hidden) = (batch, hidden)
        drug_agg = torch.mm(incidence.t(), drug_feat)
        num_drugs = incidence.sum(dim=0, keepdim=True).t().clamp(min=1)
        drug_agg = drug_agg / num_drugs

        # FiLM conditioning: gamma * drug + beta
        gamma = self.se_to_gamma(se_emb)  # (batch, hidden)
        beta = self.se_to_beta(se_emb)    # (batch, hidden)

        # Apply modulation
        conditioned = gamma * drug_agg + beta
        conditioned = self.norm(conditioned)
        conditioned = F.gelu(conditioned)
        conditioned = self.dropout(conditioned)

        return conditioned


class DrugImportanceScorer(nn.Module):
    """
    Learn which drugs are important for a specific SE.
    Outputs attention weights over drugs.
    """

    def __init__(self, drug_dim: int, se_dim: int, hidden_dim: int):
        super().__init__()

        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.se_proj = nn.Linear(se_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, drug_emb: torch.Tensor, se_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_emb: (num_drugs, drug_dim)
            se_emb: (batch_size, se_dim)
            incidence: (num_drugs, batch_size)
        Returns:
            (batch_size, hidden_dim) - importance-weighted drug features
        """
        num_drugs = drug_emb.shape[0]
        batch_size = se_emb.shape[0]

        # Project drugs and SE
        drug_proj = self.drug_proj(drug_emb)  # (num_drugs, hidden)
        se_proj = self.se_proj(se_emb)        # (batch, hidden)

        # Compute importance scores: drug-SE compatibility
        # score[d, b] = drug_proj[d] · se_proj[b]
        # (num_drugs, hidden) @ (hidden, batch) = (num_drugs, batch)
        scores = torch.mm(drug_proj, se_proj.t())  # (num_drugs, batch)

        # Mask out drugs not in sample
        scores = scores.t()  # (batch, num_drugs)
        mask = incidence.t()  # (batch, num_drugs)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get importance weights
        importance = F.softmax(scores, dim=-1)  # (batch, num_drugs)
        importance = torch.nan_to_num(importance, nan=0.0)

        # Weighted aggregation
        # (batch, num_drugs) @ (num_drugs, hidden) = (batch, hidden)
        weighted_drugs = torch.mm(importance, drug_proj)

        return weighted_drugs


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention between drug features and SE features.
    Bidirectional: drugs attend to SE, SE attends to drugs.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Drug -> SE attention
        self.drug_q = nn.Linear(dim, dim)
        self.se_kv = nn.Linear(dim, dim * 2)

        # SE -> Drug attention
        self.se_q = nn.Linear(dim, dim)
        self.drug_kv = nn.Linear(dim, dim * 2)

        self.out_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_feat: torch.Tensor, se_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_feat: (batch, dim)
            se_feat: (batch, dim)
        Returns:
            (batch, dim) - cross-attended features
        """
        batch_size = drug_feat.shape[0]

        # Drug attends to SE
        q1 = self.drug_q(drug_feat)  # (batch, dim)
        kv1 = self.se_kv(se_feat)    # (batch, dim*2)
        k1, v1 = kv1.chunk(2, dim=-1)

        # Simple attention (no multi-head for efficiency)
        attn1 = torch.sum(q1 * k1, dim=-1, keepdim=True) / self.scale
        attn1 = torch.sigmoid(attn1)
        drug_to_se = attn1 * v1

        # SE attends to Drug
        q2 = self.se_q(se_feat)
        kv2 = self.drug_kv(drug_feat)
        k2, v2 = kv2.chunk(2, dim=-1)

        attn2 = torch.sum(q2 * k2, dim=-1, keepdim=True) / self.scale
        attn2 = torch.sigmoid(attn2)
        se_to_drug = attn2 * v2

        # Combine
        combined = torch.cat([drug_to_se, se_to_drug], dim=-1)
        out = self.out_proj(combined)
        out = self.norm(out)
        out = self.dropout(out)

        return out


class GatedFusion(nn.Module):
    """
    Gated fusion of multiple feature streams.
    Learns to weight different features adaptively.
    """

    def __init__(self, dim: int, num_inputs: int, dropout: float = 0.1):
        super().__init__()
        self.num_inputs = num_inputs

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(dim * num_inputs, dim),
            nn.ReLU(),
            nn.Linear(dim, num_inputs),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *features) -> torch.Tensor:
        """
        Args:
            features: tuple of (batch, dim) tensors
        Returns:
            (batch, dim)
        """
        # Stack features: (batch, num_inputs, dim)
        stacked = torch.stack(features, dim=1)

        # Compute gates
        concat = torch.cat(features, dim=-1)  # (batch, dim * num_inputs)
        gates = self.gate(concat)  # (batch, num_inputs)
        gates = gates.unsqueeze(-1)  # (batch, num_inputs, 1)

        # Weighted sum
        fused = (stacked * gates).sum(dim=1)  # (batch, dim)

        fused = self.out_proj(fused)
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class DrugPairInteraction(nn.Module):
    """
    Fast drug pair interaction using outer product.
    Models pairwise drug interactions.
    """

    def __init__(self, drug_dim: int, hidden_dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()

        self.proj_a = nn.Linear(drug_dim, rank)
        self.proj_b = nn.Linear(drug_dim, rank)
        self.out = nn.Linear(rank, hidden_dim)
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
        # Low-rank projections
        a = self.proj_a(drug_emb)  # (num_drugs, rank)
        b = self.proj_b(drug_emb)  # (num_drugs, rank)

        # Weighted sums
        sum_a = torch.mm(incidence.t(), a)  # (batch, rank)
        sum_b = torch.mm(incidence.t(), b)  # (batch, rank)
        sum_ab = torch.mm(incidence.t(), a * b)  # (batch, rank)

        # Pairwise interaction: sum over pairs
        pairwise = sum_a * sum_b - sum_ab

        # Normalize by number of pairs
        n = incidence.sum(dim=0).clamp(min=1)
        n_pairs = (n * (n - 1) / 2).clamp(min=1).unsqueeze(1)
        pairwise = pairwise / n_pairs

        out = self.out(pairwise)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out


class HGNNv5(nn.Module):
    """
    HGNN v5: Cross-Modal Drug-SE Interaction Network

    Architecture:
    1. SE-conditioned drug encoding (FiLM)
    2. Drug importance scoring (SE-guided attention)
    3. Drug pair interaction
    4. Cross-attention between drug and SE features
    5. Gated fusion of all features
    6. Classification

    Key differences from previous versions:
    - SE actively guides drug encoding from the start
    - Multiple complementary drug representations
    - Gated fusion learns optimal feature combination
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

        # SE embeddings
        self.se_emb = nn.Embedding(num_se, se_emb_dim)

        # Drug projection
        self.drug_proj = nn.Sequential(
            nn.Linear(actual_drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # SE projection
        self.se_proj = nn.Sequential(
            nn.Linear(se_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Feature extractors
        self.se_conditioned_encoder = SEConditionedDrugEncoder(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )

        self.importance_scorer = DrugImportanceScorer(
            hidden_dim, hidden_dim, hidden_dim
        )

        self.pair_interaction = DrugPairInteraction(
            hidden_dim, hidden_dim, rank=32, dropout=dropout
        )

        # Cross-attention
        self.cross_attention = CrossAttentionLayer(
            hidden_dim, num_heads, dropout
        )

        # Simple mean aggregation for comparison
        self.mean_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gated fusion of all drug features
        self.drug_fusion = GatedFusion(hidden_dim, num_inputs=4, dropout=dropout)

        # Final fusion with SE
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
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
        # Get embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        se_emb = self.se_emb(se_indices)  # (batch, se_emb_dim)

        # Project embeddings
        drug_feat = self.drug_proj(drug_emb)  # (num_drugs, hidden)
        se_feat = self.se_proj(se_emb)        # (batch, hidden)

        # 1. SE-conditioned drug encoding
        drug_conditioned = self.se_conditioned_encoder(drug_feat, se_feat, incidence_matrix)

        # 2. Drug importance scoring
        drug_importance = self.importance_scorer(drug_feat, se_feat, incidence_matrix)

        # 3. Drug pair interaction
        drug_pairs = self.pair_interaction(drug_feat, incidence_matrix)

        # 4. Simple mean aggregation
        drug_mean = torch.mm(incidence_matrix.t(), drug_feat)
        n_drugs = incidence_matrix.sum(dim=0, keepdim=True).t().clamp(min=1)
        drug_mean = drug_mean / n_drugs
        drug_mean = self.mean_aggregator(drug_mean)

        # 5. Gated fusion of drug features
        drug_fused = self.drug_fusion(drug_conditioned, drug_importance, drug_pairs, drug_mean)

        # 6. Cross-attention with SE
        cross_feat = self.cross_attention(drug_fused, se_feat)

        # 7. Final fusion
        combined = torch.cat([drug_fused, cross_feat], dim=-1)
        fused = self.final_fusion(combined)

        # 8. Classification
        logits = self.classifier(fused)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
