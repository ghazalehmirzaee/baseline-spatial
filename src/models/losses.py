# src/models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class SpatialConsistencyLoss(nn.Module):
    """
    Loss for enforcing spatial consistency between disease predictions.
    L_spatial = Σ(w_ij * ||f_i - f_j||²)
    """

    def __init__(self):
        super().__init__()
        # Initialize anatomical region weights
        self.region_weights = {
            'upper_lung': 1.0,
            'middle_lung': 1.0,
            'lower_lung': 1.2,
            'cardiac': 1.5,
            'costophrenic': 1.3,
            'hilar': 1.1,
            'mediastinal': 1.4
        }

    def forward(
            self,
            features: torch.Tensor,
            bb_coords: torch.Tensor,
            attention: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_diseases, _ = features.shape
        loss = 0.0

        # Compute pairwise feature distances
        for i in range(num_diseases):
            for j in range(i + 1, num_diseases):
                # Get bounding box coordinates for diseases i and j
                bb_i = bb_coords[:, i]
                bb_j = bb_coords[:, j]

                # Compute spatial weight based on anatomical regions
                w_ij = self._compute_spatial_weight(bb_i, bb_j)

                # Compute feature distance
                dist = torch.norm(features[:, i] - features[:, j], dim=1)

                # Weight the distance by attention and spatial weights
                weighted_dist = w_ij * attention[:, i, j] * dist

                loss += weighted_dist.mean()

        return loss / (num_diseases * (num_diseases - 1) / 2)

    def _compute_spatial_weight(
            self,
            bb1: torch.Tensor,
            bb2: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial weight based on bounding box locations."""
        # Extract coordinates
        x1, y1, w1, h1 = bb1.unbind(-1)
        x2, y2, w2, h2 = bb2.unbind(-1)

        # Compute centers
        center1_y = y1 + h1 / 2
        center2_y = y2 + h2 / 2

        # Determine anatomical regions
        weight = torch.ones_like(x1)

        # Apply weights based on vertical position (approximate regions)
        upper_mask = (center1_y < 400) & (center2_y < 400)
        middle_mask = (center1_y >= 400) & (center1_y < 600) & \
                      (center2_y >= 400) & (center2_y < 600)
        lower_mask = (center1_y >= 600) & (center2_y >= 600)

        weight[upper_mask] *= self.region_weights['upper_lung']
        weight[middle_mask] *= self.region_weights['middle_lung']
        weight[lower_mask] *= self.region_weights['lower_lung']

        return weight


class AnatomicalConstraintLoss(nn.Module):
    """
    Loss for enforcing anatomical constraints.
    L_anatomical = Σ(R_k * ||P_k - A_k||²)
    """

    def __init__(self):
        super().__init__()
        self.anatomical_priors = self._initialize_anatomical_priors()

    def _initialize_anatomical_priors(self) -> Dict[str, torch.Tensor]:
        """Initialize anatomical prior probabilities for each disease."""
        # Define prior probabilities for each anatomical region
        priors = {
            'Atelectasis': [0.3, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],  # Upper, Middle, Lower
            'Cardiomegaly': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Cardiac
            'Effusion': [0.0, 0.0, 0.4, 0.0, 0.6, 0.0, 0.0],  # Lower, Costophrenic
            'Infiltration': [0.3, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0],  # All lung fields
            'Mass': [0.3, 0.3, 0.2, 0.0, 0.0, 0.1, 0.1],  # Various
            'Nodule': [0.4, 0.3, 0.2, 0.0, 0.0, 0.1, 0.0],  # Mostly upper
            'Pneumonia': [0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0],  # Lower predominant
            'Pneumothorax': [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],  # All lung fields
            'Consolidation': [0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0],  # Lower predominant
            'Edema': [0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0],  # Lower predominant
            'Emphysema': [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],  # Upper predominant
            'Fibrosis': [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],  # All lung fields
            'Pleural_Thickening': [0.3, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],  # All lung fields
            'Hernia': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Mediastinal
        }

        return {k: torch.tensor(v) for k, v in priors.items()}

    def forward(
            self,
            features: torch.Tensor,
            attention: torch.Tensor,
            logits: torch.Tensor
    ) -> torch.Tensor:
        batch_size = features.shape[0]
        loss = 0.0

        # Get predicted probabilities
        probs = torch.sigmoid(logits)

        for disease, prior in self.anatomical_priors.items():
            disease_idx = list(self.anatomical_priors.keys()).index(disease)

            # Get predicted disease distribution across regions
            pred_dist = attention[:, disease_idx]

            # Compute weighted difference from anatomical prior
            prior = prior.to(features.device)
            diff = torch.norm(pred_dist - prior[None, :], dim=1)

            # Weight by disease probability
            weighted_diff = probs[:, disease_idx] * diff

            loss += weighted_diff.mean()

        return loss / len(self.anatomical_priors)


class CrossDiseaseConsistencyLoss(nn.Module):
    """
    Loss for enforcing consistency between related diseases.
    L_consistency = Σ(c_ij * KL(p_i||p_j))
    """

    def __init__(self):
        super().__init__()
        self.disease_relationships = self._initialize_disease_relationships()

    def _initialize_disease_relationships(self) -> Dict[tuple, float]:
        """Initialize disease relationship confidence scores."""
        return {
            ('Effusion', 'Pleural_Thickening'): 0.8,
            ('Infiltration', 'Pneumonia'): 0.7,
            ('Mass', 'Nodule'): 0.6,
            ('Edema', 'Effusion'): 0.7,
            ('Emphysema', 'Fibrosis'): 0.5,
            ('Pneumonia', 'Consolidation'): 0.8
        }

    def forward(
            self,
            logits: torch.Tensor,
            features: torch.Tensor
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        probs = torch.sigmoid(logits)
        loss = 0.0

        disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        for (disease1, disease2), confidence in self.disease_relationships.items():
            idx1 = disease_names.index(disease1)
            idx2 = disease_names.index(disease2)

            # Compute KL divergence between disease probabilities
            p1 = probs[:, idx1]
            p2 = probs[:, idx2]

            kl_div = F.kl_div(
                torch.log(p1 + 1e-10),
                p2,
                reduction='none'
            )

            # Weight by confidence score
            weighted_kl = confidence * kl_div

            # Add feature consistency
            feature_dist = torch.norm(
                features[:, idx1] - features[:, idx2],
                dim=1
            )

            loss += (weighted_kl + confidence * feature_dist).mean()

        return loss / len(self.disease_relationships)

    