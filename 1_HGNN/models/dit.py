"""
Drug Interaction Transformer (DIT) for DDI side effect prediction.

Novel architecture that:
1. Only operates on drugs in the current sample (not all drugs globally)
2. Uses SE embedding to guide drug interaction modeling
3. Captures explicit pairwise and higher-order drug interactions
4. Combines self-attention with bilinear fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class DrugSelfAttention(nn.Module):
    """
    Self-attention among drugs in a combination.
    Captures pairwise and higher-order drug interactions.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_drugs, embed_dim)
            mask: (batch_size, num_drugs) - 1 for valid, 0 for padding
        Returns:
            (batch_size, num_drugs, embed_dim)
        """
        batch_size, num_drugs, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_drugs, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_drugs, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_drugs, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, num_drugs, num_drugs)

        # Apply mask if provided
        if mask is not None:
            # Create attention mask: (batch, 1, 1, num_drugs)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, num_drugs, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_drugs, self.embed_dim)
        out = self.out_proj(out)

        return out


class SEGuidedAttention(nn.Module):
    """
    Use SE embedding to guide attention over drugs.
    SE acts as query, drugs act as keys/values.
    """

    def __init__(self, drug_dim: int, se_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        # Project SE to query
        self.se_proj = nn.Linear(se_dim, hidden_dim)
        # Project drugs to key and value
        self.drug_k_proj = nn.Linear(drug_dim, hidden_dim)
        self.drug_v_proj = nn.Linear(drug_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        drug_emb: torch.Tensor,
        se_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            drug_emb: (batch_size, num_drugs, drug_dim)
            se_emb: (batch_size, se_dim)
            mask: (batch_size, num_drugs)
        Returns:
            (batch_size, hidden_dim) - SE-aware drug representation
        """
        batch_size, num_drugs, _ = drug_emb.shape

        # SE as query: (batch, 1, hidden_dim) -> (batch, heads, 1, head_dim)
        q = self.se_proj(se_emb).unsqueeze(1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Drugs as key/value
        k = self.drug_k_proj(drug_emb).view(batch_size, num_drugs, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.drug_v_proj(drug_emb).view(batch_size, num_drugs, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, 1, num_drugs)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)  # (batch, heads, 1, head_dim)
        out = out.squeeze(2).view(batch_size, self.hidden_dim)
        out = self.out_proj(out)

        return out


class PairwiseInteraction(nn.Module):
    """
    Explicitly model pairwise drug interactions.
    Uses bilinear layer to capture drug-drug interactions.
    """

    def __init__(self, drug_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Bilinear interaction
        self.bilinear = nn.Bilinear(drug_dim, drug_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            drug_emb: (batch_size, num_drugs, drug_dim)
            mask: (batch_size, num_drugs)
        Returns:
            (batch_size, hidden_dim) - Pairwise interaction features
        """
        batch_size, num_drugs, drug_dim = drug_emb.shape

        if num_drugs < 2:
            # Single drug, no pairwise interaction
            return torch.zeros(batch_size, self.bilinear.out_features, device=drug_emb.device)

        # Compute all pairwise interactions
        # For efficiency, use broadcasting: (batch, n, 1, dim) x (batch, 1, n, dim)
        interactions = []
        for i in range(num_drugs):
            for j in range(i + 1, num_drugs):
                inter = self.bilinear(drug_emb[:, i, :], drug_emb[:, j, :])
                interactions.append(inter)

        if len(interactions) == 0:
            return torch.zeros(batch_size, self.bilinear.out_features, device=drug_emb.device)

        # Stack and aggregate
        interactions = torch.stack(interactions, dim=1)  # (batch, num_pairs, hidden)

        # Mean pooling over pairs
        out = interactions.mean(dim=1)
        out = self.norm(out)
        out = self.dropout(out)

        return out


class DrugInteractionBlock(nn.Module):
    """
    One block of drug interaction modeling.
    Combines self-attention with feedforward.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DrugSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), mask)
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        return x


class MultiScalePooling(nn.Module):
    """
    Pool drug features at multiple scales.
    """

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        # Mean, max, and attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        self.proj = nn.Linear(embed_dim * 3, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_drugs, embed_dim)
            mask: (batch_size, num_drugs)
        Returns:
            (batch_size, output_dim)
        """
        if mask is not None:
            # Expand mask for broadcasting
            mask_expanded = mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded

            # Mean pooling
            mean_pool = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

            # Max pooling (set masked positions to -inf)
            x_for_max = x.clone()
            x_for_max[mask == 0] = float('-inf')
            max_pool, _ = x_for_max.max(dim=1)
            max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

            # Attention pooling
            attn_scores = self.attn_pool(x).squeeze(-1)  # (batch, num_drugs)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
            attn_pool = (x * attn_weights).sum(dim=1)
        else:
            mean_pool = x.mean(dim=1)
            max_pool, _ = x.max(dim=1)
            attn_scores = self.attn_pool(x).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
            attn_pool = (x * attn_weights).sum(dim=1)

        # Concatenate and project
        pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        out = self.proj(pooled)
        out = self.norm(out)

        return out


class BilinearFusion(nn.Module):
    """
    Bilinear fusion of drug and SE features.
    """

    def __init__(self, drug_dim: int, se_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        # Low-rank bilinear fusion for efficiency
        self.rank = 64
        self.drug_proj = nn.Linear(drug_dim, self.rank)
        self.se_proj = nn.Linear(se_dim, self.rank)
        self.out_proj = nn.Linear(self.rank, output_dim)

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_feat: torch.Tensor, se_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_feat: (batch_size, drug_dim)
            se_feat: (batch_size, se_dim)
        Returns:
            (batch_size, output_dim)
        """
        # Project to low-rank space
        drug_proj = torch.tanh(self.drug_proj(drug_feat))
        se_proj = torch.tanh(self.se_proj(se_feat))

        # Element-wise product (bilinear interaction)
        fused = drug_proj * se_proj

        # Project out
        out = self.out_proj(fused)
        out = self.norm(out)
        out = self.dropout(out)

        return out


class DIT(nn.Module):
    """
    Drug Interaction Transformer (DIT) for DDI side effect prediction.

    Architecture:
    1. Drug embeddings for drugs in the sample (not all drugs)
    2. Drug self-attention to capture interactions
    3. SE-guided attention to focus on relevant patterns
    4. Multi-scale pooling + Bilinear fusion
    5. Classification head

    Key differences from HGNN:
    - Only operates on sample-specific drugs
    - SE embedding guides drug interaction modeling
    - Explicit pairwise interaction modeling
    - Transformer-based architecture
    """

    def __init__(
        self,
        num_drugs: int,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        drug_embeddings: Optional[torch.Tensor] = None,
        max_drugs: int = 50
    ):
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.hidden_dim = hidden_dim
        self.max_drugs = max_drugs

        # Drug embeddings
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_dim = drug_emb_dim

        # SE embeddings
        self.se_emb = nn.Embedding(num_se, se_emb_dim)

        # Project drug embeddings to hidden dim
        self.drug_proj = nn.Sequential(
            nn.Linear(actual_drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Project SE embeddings
        self.se_proj = nn.Sequential(
            nn.Linear(se_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Drug interaction blocks (self-attention among drugs)
        self.drug_blocks = nn.ModuleList([
            DrugInteractionBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # SE-guided attention
        self.se_guided_attn = SEGuidedAttention(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)

        # Multi-scale pooling
        self.pooling = MultiScalePooling(hidden_dim, hidden_dim)

        # Pairwise interaction (for small combinations)
        self.pairwise = PairwiseInteraction(hidden_dim, hidden_dim, dropout)

        # Bilinear fusion
        self.fusion = BilinearFusion(hidden_dim * 3, hidden_dim, hidden_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
            incidence_matrix: (num_drugs, batch_size) - which drugs are in each sample
            se_indices: (batch_size,) - side effect index for each sample
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = se_indices.shape[0]
        device = se_indices.device

        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            all_drug_emb = self.drug_emb
        else:
            all_drug_emb = self.drug_emb.weight

        # Get SE embeddings
        se_feat = self.se_emb(se_indices)  # (batch, se_emb_dim)
        se_feat = self.se_proj(se_feat)  # (batch, hidden_dim)

        # For each sample, extract drugs present in that sample
        # incidence_matrix: (num_drugs, batch_size)
        # We need to handle variable-length drug sets

        # Get max number of drugs per sample for padding
        drugs_per_sample = incidence_matrix.sum(dim=0).int()  # (batch_size,)
        max_drugs_in_batch = min(drugs_per_sample.max().item(), self.max_drugs)

        # Prepare batched drug embeddings
        drug_emb_batch = torch.zeros(batch_size, max_drugs_in_batch, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_drugs_in_batch, device=device)

        for i in range(batch_size):
            # Get drug indices for this sample
            drug_indices = torch.where(incidence_matrix[:, i] > 0)[0]
            num_drugs_i = min(len(drug_indices), max_drugs_in_batch)

            if num_drugs_i > 0:
                # Get embeddings for these drugs
                sample_drug_emb = all_drug_emb[drug_indices[:num_drugs_i]]  # (num_drugs_i, drug_dim)
                sample_drug_emb = self.drug_proj(sample_drug_emb)  # (num_drugs_i, hidden_dim)

                drug_emb_batch[i, :num_drugs_i, :] = sample_drug_emb
                mask[i, :num_drugs_i] = 1

        # Drug self-attention
        drug_feat = drug_emb_batch
        for block in self.drug_blocks:
            drug_feat = block(drug_feat, mask)

        # Multi-scale pooling
        pooled_feat = self.pooling(drug_feat, mask)  # (batch, hidden_dim)

        # SE-guided attention
        se_guided_feat = self.se_guided_attn(drug_feat, se_feat, mask)  # (batch, hidden_dim)

        # Pairwise interactions (for small combinations)
        pairwise_feat = self.pairwise(drug_feat, mask)  # (batch, hidden_dim)

        # Combine drug features
        drug_combined = torch.cat([pooled_feat, se_guided_feat, pairwise_feat], dim=-1)  # (batch, hidden_dim*3)

        # Bilinear fusion with SE
        fused = self.fusion(drug_combined, se_feat)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(fused)  # (batch, num_classes)

        return logits

    def predict(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return F.softmax(logits, dim=-1)

    def predict_labels(self, incidence_matrix: torch.Tensor, se_indices: torch.Tensor) -> torch.Tensor:
        logits = self.forward(incidence_matrix, se_indices)
        return torch.argmax(logits, dim=-1)
