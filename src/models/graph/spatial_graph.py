# src/models/graph/spatial_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SpatialDistanceGraph(nn.Module):
    def __init__(
            self,
            num_diseases: int = 14,
            feature_dim: int = 768,
            hidden_dim: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1,
            anatomical_weights: Optional[torch.Tensor] = None,
            grid_sizes: List[int] = [5, 15, 25]
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.grid_sizes = grid_sizes

        # Disease embedding layers
        self.disease_embedding = nn.Linear(feature_dim, hidden_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(3)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * num_heads, feature_dim)

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Initialize anatomical weights
        if anatomical_weights is None:
            self.register_buffer("anatomical_weights", self._initialize_anatomical_weights())
        else:
            self.register_buffer("anatomical_weights", anatomical_weights)

    def _initialize_anatomical_weights(self) -> torch.Tensor:
        """Initialize anatomical region weights."""
        weights = torch.tensor([
            1.0,  # Upper lung fields
            1.0,  # Middle lung fields
            1.2,  # Lower lung fields
            1.5,  # Cardiac region
            1.3,  # Costophrenic angles
            1.1,  # Hilar regions
            1.4  # Mediastinal region
        ])
        return weights

    def compute_spatial_distribution(
            self,
            bb_coords: torch.Tensor,
            grid_size: int
    ) -> torch.Tensor:
        """Compute spatial distribution on grid for bounding boxes."""
        batch_size = bb_coords.shape[0]
        grid = torch.zeros(batch_size, self.num_diseases, grid_size, grid_size, device=bb_coords.device)

        # Scale coordinates to grid size
        scale = grid_size / 1000.0  # assuming original image is 1000x1000
        bb_coords = bb_coords * scale

        for b in range(batch_size):
            for d in range(self.num_diseases):
                if torch.any(bb_coords[b, d] != 0):  # Valid BB exists
                    x, y, w, h = bb_coords[b, d]
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(grid_size - 1, int(x + w)), min(grid_size - 1, int(y + h))
                    grid[b, d, y1:y2 + 1, x1:x2 + 1] = 1.0

        return grid.view(batch_size, self.num_diseases, -1)  # Flatten spatial dimensions

    def compute_emd(
            self,
            dist1: torch.Tensor,
            dist2: torch.Tensor,
            grid_size: int
    ) -> torch.Tensor:
        """Compute Earth Mover's Distance between spatial distributions."""
        batch_size = dist1.shape[0]
        n = grid_size * grid_size

        # Create cost matrix based on grid positions
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        pos = torch.stack([x.flatten(), y.flatten()], dim=1).float().to(dist1.device)

        C = torch.cdist(pos, pos)  # n x n cost matrix
        C = C / C.max()  # Normalize costs

        # Apply anatomical weights based on grid regions
        region_weights = self._get_region_weights(grid_size)
        C = C * region_weights

        # Prepare distributions
        P = dist1.view(batch_size, n, 1)
        Q = dist2.view(batch_size, 1, n)

        # Sinkhorn algorithm
        eps = 0.1  # regularization parameter
        max_iters = 100
        log_P = torch.log(P + 1e-8)
        log_Q = torch.log(Q + 1e-8)

        u = torch.zeros_like(P)
        v = torch.zeros_like(Q)

        K = torch.exp(-C / eps)

        for _ in range(max_iters):
            u = log_P - torch.logsumexp(v + K, dim=-1, keepdim=True)
            v = log_Q - torch.logsumexp(u + K.transpose(-2, -1), dim=-2, keepdim=True)

        # Transport plan
        P = torch.exp(u + v + K)

        # Compute EMD
        emd = torch.sum(P * C, dim=(-2, -1))

        return emd

    def _get_region_weights(self, grid_size: int) -> torch.Tensor:
        """Get region weights for the grid."""
        weights = torch.ones(grid_size, grid_size)

        # Define regions (simplified)
        upper = slice(0, grid_size // 3)
        middle = slice(grid_size // 3, 2 * grid_size // 3)
        lower = slice(2 * grid_size // 3, None)

        # Apply weights
        weights[upper, :] *= self.anatomical_weights[0]  # Upper lung
        weights[middle, :] *= self.anatomical_weights[1]  # Middle lung
        weights[lower, :] *= self.anatomical_weights[2]  # Lower lung

        return weights.flatten()

    def forward(
            self,
            features: torch.Tensor,
            bb_coords: Optional[torch.Tensor] = None,
            adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the spatial graph."""
        batch_size = features.size(0)

        # Project features to hidden dimension
        x = self.disease_embedding(features)
        x = self.norm1(x)

        # Compute spatial adjacency if BB coordinates are provided
        if bb_coords is not None:
            adj_matrix = torch.zeros(batch_size, self.num_diseases, self.num_diseases, device=features.device)

            # Multi-scale processing
            for grid_size in self.grid_sizes:
                scale_weight = 0.2 if grid_size == 5 else 0.3 if grid_size == 15 else 0.5
                dist = self.compute_spatial_distribution(bb_coords, grid_size)

                for i in range(self.num_diseases):
                    for j in range(i + 1, self.num_diseases):
                        emd = self.compute_emd(dist[:, i], dist[:, j], grid_size)
                        adj_matrix[:, i, j] += scale_weight * emd
                        adj_matrix[:, j, i] += scale_weight * emd

            # Normalize adjacency matrix
            adj_matrix = F.softmax(adj_matrix, dim=-1)
        else:
            # Use uniform adjacency if no bounding boxes
            adj_matrix = torch.ones(batch_size, self.num_diseases, self.num_diseases,
                                    device=features.device) / self.num_diseases

        # Process through GAT layers
        attention_weights = []
        for gat_layer in self.gat_layers:
            x, attn = gat_layer(x, adj_matrix)
            attention_weights.append(attn)
            x = F.elu(x)
            x = self.dropout(x)

        # Project back to original feature dimension
        output = self.output_proj(x)
        output = self.norm2(output)

        # Average attention weights across layers
        final_attention = torch.stack(attention_weights).mean(0)

        return output, final_attention


class GraphAttentionLayer(nn.Module):
    """Graph attention layer implementation."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # Linear transformations
        self.W = nn.Linear(in_features, out_features * num_heads)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, num_heads, 1)))
        nn.init.xavier_uniform_(self.a.data)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(out_features * num_heads)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)  # batch size
        N = x.size(1)  # number of nodes

        # Linear transformation
        x = self.W(x)  # [B, N, num_heads * head_dim]
        x = x.view(B, N, self.num_heads, self.head_dim)

        # Compute attention
        q = x.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        k = x.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, N]

        # Apply adjacency mask
        adj = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn = attn.masked_fill(adj == 0, float('-inf'))

        # Normalize attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        v = x.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        x = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]

        # Reshape and apply layer norm
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, num_heads, head_dim]
        x = x.view(B, N, -1)  # [B, N, num_heads * head_dim]
        x = self.norm(x)

        return x, attn.mean(dim=1)  # Average attention weights across heads

    