# src/models/fusion.py
from typing import Optional, Tuple

import torch
from torch import nn


class AnatomicalAttention(nn.Module):
    def __init__(
            self,
            num_diseases: int,
            num_regions: int,
            feature_dim: int
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_regions = num_regions
        self.feature_dim = feature_dim

        # Region embeddings
        self.region_embeddings = nn.Parameter(
            torch.randn(num_regions, feature_dim)
        )

        # Attention layers
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.attention_scale = feature_dim ** -0.5
        self.norm = nn.LayerNorm(feature_dim)


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

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        device = features.device

        # Move region embeddings to the correct device
        region_embeds = self.region_embeddings.to(device)

        # Project features
        Q = self.query_proj(features)
        K = self.key_proj(region_embeds[None].expand(batch_size, -1, -1))
        V = self.value_proj(region_embeds[None].expand(batch_size, -1, -1))

        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.attention_scale
        attention_weights = torch.nn.functional.softmax(attention, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attention_weights, V)
        attended_features = self.norm(attended_features)

        return attended_features, attention_weights


class CrossAttentionFusion(nn.Module):
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
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
            self,
            vit_features: torch.Tensor,
            graph_features: torch.Tensor,
            anatomical_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size = vit_features.size(0)
        device = vit_features.device

        # Reshape for multi-head attention
        q = self.q_proj(vit_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(graph_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(graph_features).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Combine features
        combined = torch.cat([vit_features, out, anatomical_features], dim=-1)

        # Final fusion
        fused = self.fusion(combined)
        fused = self.norm(fused)

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

