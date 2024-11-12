# src/models/graph/spatial_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SpatialDistanceGraph(nn.Module):
    """Implements the spatial distance graph component."""

    def __init__(
            self,
            num_diseases: int = 14,
            feature_dim: int = 768,
            hidden_dim: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1,
            anatomical_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Initialize anatomical region weights if not provided
        if anatomical_weights is None:
            self.register_buffer(
                "anatomical_weights",
                self._initialize_anatomical_weights()
            )
        else:
            self.register_buffer("anatomical_weights", anatomical_weights)

        # Disease embedding layers
        self.disease_embedding = nn.Linear(feature_dim, hidden_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout,
                concat=True
            ) for _ in range(3)  # Stack of 3 GAT layers
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * num_heads, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _initialize_anatomical_weights(self) -> torch.Tensor:
        """Initialize anatomical region weights based on clinical knowledge."""
        weights = {
            "upper_lung": 1.0,
            "middle_lung": 1.0,
            "lower_lung": 1.2,
            "cardiac": 1.5,
            "costophrenic": 1.3,
            "hilar": 1.1,
            "mediastinal": 1.4
        }

        # Create 7x7 weight matrix for 7 anatomical regions
        W = torch.ones(7, 7)
        for i, w1 in enumerate(weights.values()):
            for j, w2 in enumerate(weights.values()):
                W[i, j] = (w1 + w2) / 2

        return W

    def compute_spatial_distribution(
            self,
            bb_coords: torch.Tensor,
            grid_size: int
    ) -> torch.Tensor:
        """Compute spatial distribution on grid for bounding boxes."""
        batch_size = bb_coords.size(0)
        grid = torch.zeros(batch_size, self.num_diseases, grid_size, grid_size)

        # Normalize coordinates to [0, grid_size]
        bb_coords = bb_coords / 1000.0 * grid_size

        for b in range(batch_size):
            for d in range(self.num_diseases):
                if torch.any(bb_coords[b, d] != 0):  # Valid BB exists
                    x, y, w, h = bb_coords[b, d]
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(grid_size - 1, int(x + w)), min(grid_size - 1, int(y + h))
                    grid[b, d, y1:y2 + 1, x1:x2 + 1] = 1.0

        return grid

    def compute_emd(
            self,
            dist1: torch.Tensor,
            dist2: torch.Tensor,
            weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Earth Mover's Distance between distributions."""
        if weights is None:
            weights = self.anatomical_weights

        # Reshape distributions to 2D
        d1 = dist1.view(dist1.size(0), -1)
        d2 = dist2.view(dist2.size(0), -1)

        # Normalize distributions
        d1 = d1 / (d1.sum(dim=-1, keepdim=True) + 1e-6)
        d2 = d2 / (d2.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute cost matrix
        size = int(np.sqrt(d1.size(-1)))
        xx, yy = torch.meshgrid(torch.arange(size), torch.arange(size))
        C = torch.sqrt((xx[None] - xx[:, None]) ** 2 + (yy[None] - yy[:, None]) ** 2).to(d1.device)
        C = C * weights.view(-1, 1)

        # Sinkhorn algorithm for EMD approximation
        P = self._sinkhorn(d1, d2, C, epsilon=0.1, niter=100)

        return torch.sum(P * C, dim=(-2, -1))

    def _sinkhorn(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            C: torch.Tensor,
            epsilon: float,
            niter: int
    ) -> torch.Tensor:
        """Sinkhorn algorithm for optimal transport."""
        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # K is the kernel matrix
        K = torch.exp(-C / epsilon)

        for _ in range(niter):
            u = epsilon * (torch.log(a + 1e-8) - torch.logsumexp(v[:, None] + K / epsilon, dim=-1))
            v = epsilon * (torch.log(b + 1e-8) - torch.logsumexp(u[:, None] + K.transpose(-2, -1) / epsilon, dim=-1))

        # Return transport plan
        P = torch.exp((u[:, None] + v[:, None] + K) / epsilon)
        return P

    def forward(
            self,
            features: torch.Tensor,
            bb_coords: Optional[torch.Tensor] = None,
            adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the spatial graph module.

        Args:
            features: Input features from ViT [batch_size, num_diseases, feature_dim]
            bb_coords: Bounding box coordinates [batch_size, num_diseases, 4]
            adj_matrix: Pre-computed adjacency matrix [num_diseases, num_diseases]

        Returns:
            Updated features and attention weights
        """
        batch_size = features.size(0)

        # Project features to hidden dimension
        x = self.disease_embedding(features)

        # Compute spatial distributions if BB coordinates are provided
        if bb_coords is not None:
            # Multi-scale processing
            distributions = []
            for grid_size in [5, 15, 25]:
                dist = self.compute_spatial_distribution(bb_coords, grid_size)
                distributions.append(dist)

            # Compute EMD-based adjacency matrix if not provided
            if adj_matrix is None:
                adj_matrix = torch.zeros(batch_size, self.num_diseases, self.num_diseases)

                for scale, dist in enumerate(distributions):
                    scale_weight = [0.2, 0.3, 0.5][scale]
                    for i in range(self.num_diseases):
                        for j in range(i + 1, self.num_diseases):
                            emd = self.compute_emd(dist[:, i], dist[:, j])
                            adj_matrix[:, i, j] += scale_weight * emd
                            adj_matrix[:, j, i] += scale_weight * emd

                # Normalize adjacency matrix
                adj_matrix = F.softmax(adj_matrix, dim=-1)

        # Process through GAT layers
        attention_weights = {}
        for i, gat_layer in enumerate(self.gat_layers):
            x, attn = gat_layer(x, adj_matrix)
            attention_weights[f'layer_{i}'] = attn
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)

        # Project back to original feature dimension
        output = self.output_proj(x)

        return output, attention_weights


class GraphAttentionLayer(nn.Module):
    """Implementation of Graph Attention Layer."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float,
            concat: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations
        self.W = nn.Parameter(
            torch.zeros(size=(in_features, num_heads * out_features))
        )
        nn.init.xavier_uniform_(self.W.data)

        # Attention mechanisms
        self.a = nn.Parameter(
            torch.zeros(size=(2 * out_features, num_heads, 1))
        )
        nn.init.xavier_uniform_(self.a.data)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GAT layer.

        Args:
            x: Input features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Updated node features and attention coefficients
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # Linear transformation
        Wh = torch.matmul(x, self.W)
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, -1)

        # Compute attention coefficients
        Whj = Wh.unsqueeze(2)
        Whi = Wh.unsqueeze(1)

        # Concatenate features
        concat_features = torch.cat([Whj.repeat(1, 1, num_nodes, 1, 1),
                                     Whi.repeat(1, num_nodes, 1, 1, 1)], dim=-1)

        # Compute attention scores
        e = torch.matmul(concat_features, self.a).squeeze(-1)

        # Mask attention scores based on adjacency matrix
        if adj is not None:
            adj = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
            e = e.masked_fill(adj == 0, float('-inf'))

        # Apply attention
        attention = F.softmax(e, dim=2)
        attention = self.dropout_layer(attention)

        # Apply attention to values
        h_prime = torch.matmul(
            attention.permute(0, 3, 1, 2),
            Wh.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        if self.concat:
            # Concatenate multi-head results
            out = h_prime.reshape(batch_size, num_nodes, -1)
        else:
            # Average multi-head results
            out = h_prime.mean(dim=2)

        return out, attention



