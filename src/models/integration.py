# src/models/integration.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer
import math


class IntegratedModel(nn.Module):
    """
    Integrates pretrained ViT with Spatial Distance Graph
    """

    def __init__(
            self,
            pretrained_path: str,
            num_classes: int = 14,
            freeze_vit: bool = True,
            feature_dim: int = 768,
            graph_hidden_dim: int = 256,
            graph_num_heads: int = 8,
            anatomical_regions: int = 7
    ):
        super().__init__()

        # Load pretrained ViT
        self.vit = self._load_pretrained_vit(pretrained_path)

        if freeze_vit:
            self._freeze_vit_layers()

        # Initialize spatial graph component
        self.spatial_graph = SpatialDistanceGraph(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=graph_hidden_dim,
            num_heads=graph_num_heads
        )

        # Initialize anatomical attention
        self.anatomical_attention = AnatomicalAttention(
            num_diseases=num_classes,
            num_regions=anatomical_regions,
            feature_dim=feature_dim
        )

        # Initialize fusion layer
        self.fusion = CrossAttentionFusion(
            feature_dim=feature_dim,
            num_heads=8
        )

        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )

        # Loss components
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.spatial_loss = SpatialConsistencyLoss()
        self.anatomical_loss = AnatomicalConstraintLoss()
        self.consistency_loss = CrossDiseaseConsistencyLoss()

        # Loss weights
        self.loss_weights = {
            'cls': 1.0,
            'spatial': 0.3,
            'anatomical': 0.2,
            'consistency': 0.1
        }

    def _load_pretrained_vit(self, checkpoint_path: str) -> nn.Module:
        """Load pretrained ViT from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        # Initialize ViT model
        vit = VisionTransformer(
            img_size=1000,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_classes=14,
            depth=12
        )

        # Load state dict
        vit.load_state_dict(checkpoint['model_state_dict'])
        return vit

    def _freeze_vit_layers(self, unfreeze_last_n: int = 0):
        """Freeze ViT layers except last n blocks if specified."""
        for param in self.vit.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # Unfreeze last n transformer blocks
            for block in self.vit.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Unfreeze final norm and head
            for param in self.vit.norm.parameters():
                param.requires_grad = True

    def unfreeze_vit_layers(self, num_layers: int):
        """Unfreeze specified number of last ViT layers."""
        self._freeze_vit_layers(unfreeze_last_n=num_layers)

    def forward(
            self,
            images: torch.Tensor,
            bb_coords: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the integrated model.

        Args:
            images: Input images [batch_size, channels, height, width]
            bb_coords: Bounding box coordinates [batch_size, num_diseases, 4]
            labels: Ground truth labels [batch_size, num_diseases]

        Returns:
            Dictionary containing model outputs and losses
        """
        batch_size = images.shape[0]

        # Get ViT features
        vit_features = self.vit.forward_features(images)  # [B, N, D]
        cls_token = vit_features[:, 0]  # Use CLS token

        # Process through spatial graph
        graph_features, spatial_attn = self.spatial_graph(
            cls_token,
            bb_coords
        )

        # Apply anatomical attention
        anatomical_features, anatomical_attn = self.anatomical_attention(
            graph_features
        )

        # Fuse features
        fused_features = self.fusion(
            cls_token,
            graph_features,
            anatomical_features
        )

        # Final classification
        logits = self.classifier(fused_features)

        outputs = {
            'logits': logits,
            'vit_features': cls_token,
            'graph_features': graph_features,
            'anatomical_features': anatomical_features,
            'spatial_attention': spatial_attn,
            'anatomical_attention': anatomical_attn
        }

        # Compute losses if labels are provided
        if labels is not None:
            losses = self._compute_losses(
                logits=logits,
                labels=labels,
                vit_features=cls_token,
                graph_features=graph_features,
                anatomical_features=anatomical_features,
                bb_coords=bb_coords,
                spatial_attn=spatial_attn,
                anatomical_attn=anatomical_attn
            )
            outputs.update(losses)

        return outputs

    def _compute_losses(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            vit_features: torch.Tensor,
            graph_features: torch.Tensor,
            anatomical_features: torch.Tensor,
            bb_coords: Optional[torch.Tensor],
            spatial_attn: torch.Tensor,
            anatomical_attn: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""

        # Classification loss
        cls_loss = self.bce_loss(logits, labels).mean()

        # Spatial consistency loss
        spatial_loss = self.spatial_loss(
            features=graph_features,
            bb_coords=bb_coords,
            attention=spatial_attn
        ) if bb_coords is not None else torch.tensor(0.0).to(logits.device)

        # Anatomical constraint loss
        anatomical_loss = self.anatomical_loss(
            features=anatomical_features,
            attention=anatomical_attn,
            logits=logits
        )

        # Cross-disease consistency loss
        consistency_loss = self.consistency_loss(
            logits=logits,
            features=graph_features
        )

        # Combine losses
        total_loss = (
                self.loss_weights['cls'] * cls_loss +
                self.loss_weights['spatial'] * spatial_loss +
                self.loss_weights['anatomical'] * anatomical_loss +
                self.loss_weights['consistency'] * consistency_loss
        )

        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'spatial_loss': spatial_loss,
            'anatomical_loss': anatomical_loss,
            'consistency_loss': consistency_loss
        }
