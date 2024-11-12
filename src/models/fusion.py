# src/models/fusion.py
from typing import Optional, Tuple

import torch
from torch import nn


class AnatomicalAttention(nn.Module):
    """
    Attention mechanism based on anatomical regions.
    """

    def __init__(
            self,
            num_diseases: int,
            num_regions: int,
            feature_dim: int
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.num_regions = num_regions

        # Region embeddings
        self.region_embeddings = nn.Parameter(
            torch.randn(num_regions, feature_dim)
        )

        # Attention layers
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.attention_scale = feature_dim ** -0.5

        # Region weights based on clinical knowledge
        self.register_buffer(
            'region_weights',
            self._initialize_region_weights()
        )

    def _initialize_region_weights(self) -> torch.Tensor:
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

    def forward(
            self,
            features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply anatomical attention to features.

        Args:
            features: Input features [batch_size, num_diseases, feature_dim]

        Returns:
            Tuple of (attended features, attention weights)
        """
        batch_size = features.shape[0]

        # Project queries, keys, values
        Q = self.query_proj(features)
        K = self.key_proj(self.region_embeddings[None].expand(batch_size, -1, -1))
        V = self.value_proj(self.region_embeddings[None].expand(batch_size, -1, -1))

        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_scale

        # Apply region weights
        attention = attention * self.region_weights[None, None]

        # Normalize attention weights
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        attended_features = torch.matmul(attention, V)

        return attended_features, attention


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based feature fusion.
    """

    def __init__(
            self,
            feature_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention layers
        self.vit_to_graph = MultiHeadAttention(
            feature_dim,
            num_heads,
            dropout
        )

        self.graph_to_vit = MultiHeadAttention(
            feature_dim,
            num_heads,
            dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            vit_features: torch.Tensor,
            graph_features: torch.Tensor,
            anatomical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse different feature types using cross-attention.
        """
        # Cross-attention between ViT and graph features
        vit_attended = self.vit_to_graph(
            query=vit_features,
            key=graph_features,
            value=graph_features
        )

        graph_attended = self.graph_to_vit(
            query=graph_features,
            key=vit_features,
            value=vit_features
        )

        # Concatenate all features
        combined = torch.cat([
            vit_features,
            graph_attended,
            anatomical_features
        ], dim=-1)

        # Fuse features
        fused = self.fusion(combined)

        return fused


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    """

    def __init__(
            self,
            feature_dim: int,
            num_heads: int,
            dropout: float
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        # Project and reshape
        q = self.q_proj(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.feature_dim
        )

        return self.o_proj(out)

