"""
Graph Neural Network models for drug-drug interaction side effect prediction.
Supports GAT (Graph Attention Network) and GCN (Graph Convolutional Network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from typing import Optional, List


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for drug features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize GAT encoder.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim

            # For intermediate layers, output is heads * out_ch, then normalized
            if i < num_layers - 1:
                self.convs.append(
                    GATConv(in_ch, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                # Last layer: average heads
                self.convs.append(
                    GATConv(hidden_dim, out_ch, heads=heads, dropout=dropout, concat=False)
                )
                self.norms.append(nn.LayerNorm(out_ch))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GAT layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder for drug features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize GCN encoder.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim

            self.convs.append(GCNConv(in_ch, out_ch))
            self.norms.append(nn.LayerNorm(out_ch))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_dim)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GCN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x


class SEProjector(nn.Module):
    """Projector for side effect embeddings."""

    def __init__(
        self,
        num_se: int,
        emb_dim: int,
        out_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize SE Projector.

        Args:
            num_se: Number of unique side effects
            emb_dim: Embedding dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(num_se, emb_dim)
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, se_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            se_indices: Side effect indices (batch_size,)

        Returns:
            SE features (batch_size, out_dim)
        """
        se_emb = self.embedding(se_indices)
        return self.projector(se_emb)


class MLPClassifier(nn.Module):
    """MLP classifier for final prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize MLP Classifier.

        Args:
            in_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, in_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        return self.mlp(x)


class EdgePredictor(nn.Module):
    """
    Edge predictor that combines source and target drug features.
    """

    def __init__(
        self,
        node_dim: int,
        se_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.2,
        combine_method: str = 'concat'
    ):
        """
        Initialize Edge Predictor.

        Args:
            node_dim: Node feature dimension
            se_dim: Side effect feature dimension
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            combine_method: How to combine node features ('concat', 'hadamard', 'sum')
        """
        super().__init__()
        self.combine_method = combine_method

        if combine_method == 'concat':
            edge_dim = node_dim * 2 + se_dim
        else:  # hadamard or sum
            edge_dim = node_dim + se_dim

        self.classifier = MLPClassifier(
            in_dim=edge_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        source_emb: torch.Tensor,
        target_emb: torch.Tensor,
        se_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            source_emb: Source node embeddings (batch_size, node_dim)
            target_emb: Target node embeddings (batch_size, node_dim)
            se_emb: Side effect embeddings (batch_size, se_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        if self.combine_method == 'concat':
            edge_features = torch.cat([source_emb, target_emb, se_emb], dim=-1)
        elif self.combine_method == 'hadamard':
            edge_features = torch.cat([source_emb * target_emb, se_emb], dim=-1)
        else:  # sum
            edge_features = torch.cat([source_emb + target_emb, se_emb], dim=-1)

        return self.classifier(edge_features)


class GNN(nn.Module):
    """
    Graph Neural Network for DDI side effect prediction.

    Architecture:
    1. Drug embeddings -> GNN encoder (GAT/GCN) -> Drug node features
    2. SE indices -> Embedding + Projector -> SE features
    3. For each edge (drug pair): combine source + target + SE features
    4. Edge predictor MLP -> Binary classification
    """

    def __init__(
        self,
        num_drugs: int,
        num_se: int,
        drug_emb_dim: int = 128,
        se_emb_dim: int = 128,
        drug_out_dim: int = 64,
        se_out_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        mlp_hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.1,
        gnn_type: str = 'gat',
        combine_method: str = 'concat',
        drug_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize GNN model.

        Args:
            num_drugs: Number of unique drugs
            num_se: Number of unique side effects
            drug_emb_dim: Drug embedding dimension
            se_emb_dim: SE embedding dimension
            drug_out_dim: Drug encoder output dimension
            se_out_dim: SE projector output dimension
            num_gnn_layers: Number of GNN layers
            num_heads: Number of attention heads (for GAT)
            mlp_hidden_dims: MLP hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            gnn_type: Type of GNN ('gat' or 'gcn')
            combine_method: How to combine drug pair features
            drug_embeddings: Pre-computed drug embeddings (optional)
        """
        super().__init__()

        self.num_drugs = num_drugs
        self.num_se = num_se
        self.drug_out_dim = drug_out_dim
        self.se_out_dim = se_out_dim
        self.gnn_type = gnn_type

        # Drug embedding layer
        if drug_embeddings is not None:
            self.drug_emb = nn.Parameter(drug_embeddings, requires_grad=False)
            actual_drug_emb_dim = drug_embeddings.shape[1]
        else:
            self.drug_emb = nn.Embedding(num_drugs, drug_emb_dim)
            actual_drug_emb_dim = drug_emb_dim

        # GNN encoder
        if gnn_type == 'gat':
            self.drug_encoder = GATEncoder(
                in_dim=actual_drug_emb_dim,
                hidden_dim=drug_emb_dim,
                out_dim=drug_out_dim,
                num_layers=num_gnn_layers,
                heads=num_heads,
                dropout=dropout
            )
        else:  # gcn
            self.drug_encoder = GCNEncoder(
                in_dim=actual_drug_emb_dim,
                hidden_dim=drug_emb_dim,
                out_dim=drug_out_dim,
                num_layers=num_gnn_layers,
                dropout=dropout
            )

        # SE projector
        self.se_projector = SEProjector(
            num_se=num_se,
            emb_dim=se_emb_dim,
            out_dim=se_out_dim,
            dropout=dropout
        )

        # Edge predictor
        self.edge_predictor = EdgePredictor(
            node_dim=drug_out_dim,
            se_dim=se_out_dim,
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            combine_method=combine_method
        )

        # Simple projection for no-message-passing mode
        self.simple_proj = nn.Sequential(
            nn.Linear(actual_drug_emb_dim, drug_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(drug_emb_dim, drug_out_dim),
            nn.LayerNorm(drug_out_dim)
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        se_indices: torch.Tensor,
        graph_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            edge_index: Drug pair indices to classify (2, batch_size)
            se_indices: Side effect indices (batch_size,)
            graph_edge_index: Full graph edge index for message passing
                             If None, skips GNN and uses embeddings directly

        Returns:
            Logits (batch_size, num_classes)
        """
        # Get drug embeddings
        if isinstance(self.drug_emb, nn.Parameter):
            drug_emb = self.drug_emb
        else:
            drug_emb = self.drug_emb.weight

        # Encode drugs through GNN (if graph provided) or use embeddings directly
        if graph_edge_index is not None:
            drug_features = self.drug_encoder(drug_emb, graph_edge_index)
        else:
            # Skip message passing, use simple projection
            drug_features = self.simple_proj(drug_emb)

        # Get source and target drug features for edges to classify
        source_emb = drug_features[edge_index[0]]
        target_emb = drug_features[edge_index[1]]

        # Project side effects
        se_features = self.se_projector(se_indices)

        # Predict edges
        logits = self.edge_predictor(source_emb, target_emb, se_features)

        return logits

    def predict(
        self,
        edge_index: torch.Tensor,
        se_indices: torch.Tensor,
        graph_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict probabilities."""
        logits = self.forward(edge_index, se_indices, graph_edge_index)
        return F.softmax(logits, dim=-1)

    def predict_labels(
        self,
        edge_index: torch.Tensor,
        se_indices: torch.Tensor,
        graph_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict class labels."""
        logits = self.forward(edge_index, se_indices, graph_edge_index)
        return torch.argmax(logits, dim=-1)
