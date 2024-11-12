# src/models/integration.py

from typing import Optional, Dict

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import math

from src.models.fusion import CrossAttentionFusion, AnatomicalAttention
from src.models.graph.spatial_graph import SpatialDistanceGraph
from src.models.losses import SpatialConsistencyLoss, AnatomicalConstraintLoss, CrossDiseaseConsistencyLoss


def interpolate_pos_embed(pos_embed_checkpoint, num_patches):
    """
    Interpolate position embeddings for different image sizes.
    """
    num_extra_tokens = 1  # cls token
    embedding_dim = pos_embed_checkpoint.shape[-1]

    # Only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, int(math.sqrt(pos_tokens.shape[1])),
                                    int(math.sqrt(pos_tokens.shape[1])), embedding_dim)

    # Interpolate
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens.permute(0, 3, 1, 2),
        size=(int(math.sqrt(num_patches)), int(math.sqrt(num_patches))),
        mode='bicubic',
        align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    # Combine with cls token
    pos_embed = torch.cat((pos_embed_checkpoint[:, :num_extra_tokens], pos_tokens), dim=1)

    return pos_embed


class IntegratedModel(nn.Module):
    """Integrated model combining ViT with spatial graph attention."""

    # Define anatomical regions as class attribute
    ANATOMICAL_REGIONS = {
        'upper_lung': {'weight': 1.0, 'position': (0.25, 0.25)},
        'middle_lung': {'weight': 1.0, 'position': (0.5, 0.5)},
        'lower_lung': {'weight': 1.2, 'position': (0.75, 0.75)},
        'cardiac': {'weight': 1.5, 'position': (0.5, 0.5)},
        'costophrenic': {'weight': 1.3, 'position': (0.8, 0.8)},
        'hilar': {'weight': 1.1, 'position': (0.5, 0.3)},
        'mediastinal': {'weight': 1.4, 'position': (0.5, 0.4)}
    }

    def __init__(
            self,
            pretrained_path: str,
            num_classes: int = 14,
            freeze_vit: bool = True,
            feature_dim: int = 768,
            graph_hidden_dim: int = 256,
            graph_num_heads: int = 8,
            image_size: int = 1000,
            patch_size: int = 16
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_anatomical_regions = len(self.ANATOMICAL_REGIONS)

        # Load pretrained ViT
        self.vit = self._load_pretrained_vit(pretrained_path)

        if freeze_vit:
            self._freeze_vit_layers()

        # Initialize spatial graph component
        self.spatial_graph = SpatialDistanceGraph(
            num_diseases=num_classes,
            feature_dim=feature_dim,
            hidden_dim=graph_hidden_dim,
            num_heads=graph_num_heads,
            num_anatomical_regions=self.num_anatomical_regions
        )

        # Initialize anatomical attention
        self.anatomical_attention = AnatomicalAttention(
            num_diseases=num_classes,
            num_regions=self.num_anatomical_regions,
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
        """Load pretrained ViT with proper position embedding interpolation."""
        # Calculate number of patches for current image size
        num_patches = (self.image_size // self.patch_size) ** 2

        # Initialize ViT with current image size
        vit = VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.feature_dim,
            num_heads=12,
            num_classes=self.num_classes,
            depth=12
        )

        try:
            # Add safe globals for numpy scalar types
            torch.serialization.add_safe_globals([
                'numpy._core.multiarray.scalar',
                'numpy.core.multiarray.scalar',
                '_codecs.encode',
                '__builtin__.getattr',
            ])

            # Load checkpoint with modified settings
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False  # Changed to False to handle numpy scalars
            )

            # Get state dict (handle different checkpoint formats)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Handle position embedding interpolation
            pos_embed_checkpoint = state_dict['pos_embed']
            if pos_embed_checkpoint.shape[1] != vit.pos_embed.shape[1]:
                print(f"Interpolating position embeddings from {pos_embed_checkpoint.shape} to {vit.pos_embed.shape}")
                state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint, num_patches)

            # Load state dict
            msg = vit.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint with message: {msg}")

            return vit

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("\nDetailed error information:")
            import traceback
            traceback.print_exc()
            raise

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
        vit_features = self.vit.forward_features(images)
        cls_token = vit_features[:, 0]

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

        # Prepare outputs
        outputs = {
            'logits': logits,
            'vit_features': cls_token,
            'graph_features': graph_features,
            'anatomical_features': anatomical_features,
            'spatial_attention': spatial_attn,
            'anatomical_attention': anatomical_attn
        }

        # Compute loss if labels are provided
        if labels is not None:
            loss = self.compute_loss(
                logits=logits,
                labels=labels,
                vit_features=cls_token,
                graph_features=graph_features,
                anatomical_features=anatomical_features,
                bb_coords=bb_coords,
                spatial_attn=spatial_attn,
                anatomical_attn=anatomical_attn
            )
            outputs['loss'] = loss

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

