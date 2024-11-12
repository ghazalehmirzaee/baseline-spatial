# src/visualization/attention.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any
import cv2


class AttentionVisualizer:
    """Visualize attention maps from the model."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    def visualize(
            self,
            image: torch.Tensor,
            attention_maps: Dict[str, torch.Tensor],
            save_dir: Path,
            bbox: Optional[torch.Tensor] = None
    ) -> None:
        """
        Visualize attention maps and save visualizations.

        Args:
            image: Input image tensor [C, H, W]
            attention_maps: Dictionary of attention maps from different layers
            save_dir: Directory to save visualizations
            bbox: Optional bounding box coordinates [num_diseases, 4]
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert image to numpy and normalize
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Plot original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        if bbox is not None:
            self._plot_bounding_boxes(plt.gca(), bbox)
        plt.axis('off')
        plt.savefig(save_dir / 'original_image.png')
        plt.close()

        # Plot attention maps
        for layer_name, attn_map in attention_maps.items():
            # Average attention across heads
            if len(attn_map.shape) > 3:
                attn_map = attn_map.mean(1)

            # Reshape and upsample to image size
            h = w = int(np.sqrt(attn_map.shape[-1]))
            attn_map = attn_map.reshape(-1, h, w)
            attn_map = F.interpolate(
                attn_map.unsqueeze(0),
                size=image.shape[1:],
                mode='bilinear',
                align_corners=False
            )[0]

            # Plot each attention head
            for idx in range(attn_map.shape[0]):
                plt.figure(figsize=(10, 10))
                plt.imshow(image_np, alpha=0.5)
                plt.imshow(
                    attn_map[idx].numpy(),
                    cmap='jet',
                    alpha=0.5
                )
                if bbox is not None:
                    self._plot_bounding_boxes(plt.gca(), bbox)
                plt.axis('off')
                plt.colorbar()
                plt.savefig(
                    save_dir / f'{layer_name}_head_{idx}.png')
                plt.close()

    def _plot_bounding_boxes(
        self,
        ax: plt.Axes,
        bbox: torch.Tensor,
        colors: Optional[List[str]] = None
    ) -> None:
        """Plot bounding boxes on given axes."""
        if colors is None:
            colors = plt.cm.rainbow(np.linspace(0, 1, bbox.shape[0]))

        for i in range(bbox.shape[0]):
            if torch.any(bbox[i] > 0):
                x, y, w, h = bbox[i].numpy()
                rect = plt.Rectangle(
                    (x, y), w, h,
                    fill=False,
                    edgecolor=colors[i],
                    linewidth=2
                )
                ax.add_patch(rect)


# src/visualization/gradcam.py

class GradCAMVisualizer:
    """Generate GradCAM visualizations for model predictions."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks for GradCAM."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks on the last convolutional layer
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

        if target_layer is None:
            raise ValueError("No convolutional layer found in model")

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))

    def generate(
        self,
        image: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Generate GradCAM activation map.

        Args:
            image: Input image tensor [1, C, H, W]
            target_class: Target class index

        Returns:
            GradCAM activation map
        """
        # Forward pass
        output = self.model(image)

        if isinstance(output, dict):
            output = output['logits']

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output[0, target_class].backward()

        # Generate GradCAM
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        activations = self.activations

        weights = torch.mean(gradients, dim=1, keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)

        # Normalize
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy()

    def save_visualization(
        self,
        image: torch.Tensor,
        cam: np.ndarray,
        save_path: Path
    ) -> None:
        """Save GradCAM visualization."""
        # Convert image to numpy and normalize
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(float) / 255

        # Combine image and heatmap
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('GradCAM')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(image_np)
        plt.imshow(heatmap, alpha=0.4)
        plt.title('Combined')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# src/visualization/error_analysis.py

class ErrorAnalyzer:
    """Analyze and visualize model errors."""

    def __init__(
        self,
        disease_names: List[str],
        output_dir: Path
    ):
        self.disease_names = disease_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_errors(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.

        Args:
            preds: Model predictions [N, num_classes]
            labels: Ground truth labels [N, num_classes]
            features: Optional feature vectors [N, feature_dim]

        Returns:
            Dictionary containing error analysis results
        """
        results = {}

        # Convert predictions to binary
        binary_preds = (preds > 0.5).float()

        # Compute per-class error rates
        error_rates = {
            disease: ((binary_preds[:, i] != labels[:, i]).float().mean().item())
            for i, disease in enumerate(self.disease_names)
        }

        results['error_rates'] = error_rates

        # Analyze error co-occurrence
        error_matrix = self._compute_error_cooccurrence(binary_preds, labels)
        results['error_cooccurrence'] = error_matrix

        # Analyze false positives and negatives
        fp_fn_analysis = self._analyze_fp_fn(binary_preds, labels)
        results['fp_fn_analysis'] = fp_fn_analysis

        # Feature space analysis if features are provided
        if features is not None:
            feature_analysis = self._analyze_feature_space(
                features, binary_preds, labels
            )
            results['feature_analysis'] = feature_analysis

        # Generate visualizations
        self._plot_error_analysis(results)

        return results

    def _compute_error_cooccurrence(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """Compute error co-occurrence matrix."""
        errors = (preds != labels).float()
        error_matrix = torch.mm(errors.t(), errors) / len(errors)
        return error_matrix.numpy()

    def _analyze_fp_fn(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Analyze false positives and false negatives."""
        results = {}

        for i, disease in enumerate(self.disease_names):
            fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).float().mean().item()
            fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).float().mean().item()

            results[disease] = {
                'false_positive_rate': fp,
                'false_negative_rate': fn,
                'fp_fn_ratio': fp/fn if fn > 0 else float('inf')
            }

        return results

    def _analyze_feature_space(
        self,
        features: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze error patterns in feature space."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        results = {}

        # Apply dimensionality reduction
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)

        features_pca = pca.fit_transform(features.numpy())
        features_tsne = tsne.fit_transform(features.numpy())

        # Analyze error clusters
        error_mask = (preds != labels).any(dim=1)

        results['pca'] = {
            'coordinates': features_pca,
            'error_mask': error_mask.numpy()
        }

        results['tsne'] = {
            'coordinates': features_tsne,
            'error_mask': error_mask.numpy()
        }

        return results

    def _plot_error_analysis(self, results: Dict[str, Any]) -> None:
        """Generate and save error analysis visualizations."""
        # Plot error rates
        plt.figure(figsize=(15, 5))
        error_rates = list(results['error_rates'].values())
        plt.bar(self.disease_names, error_rates)
        plt.xticks(rotation=45)
        plt.title('Error Rates by Disease')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_rates.png')
        plt.close()

        # Plot error co-occurrence matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            results['error_cooccurrence'],
            xticklabels=self.disease_names,
            yticklabels=self.disease_names,
            cmap='YlOrRd'
        )
        plt.title('Error Co-occurrence Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_cooccurrence.png')
        plt.close()

        # Plot false positive vs false negative rates
        fp_rates = [x['false_positive_rate']
                   for x in results['fp_fn_analysis'].values()]
        fn_rates = [x['false_negative_rate']
                   for x in results['fp_fn_analysis'].values()]

        plt.figure(figsize=(10, 10))
        plt.scatter(fp_rates, fn_rates)
        for i, disease in enumerate(self.disease_names):
            plt.annotate(disease, (fp_rates[i], fn_rates[i]))
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('False Positive vs False Negative Rates')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fp_fn_analysis.png')
        plt.close()

        # Plot feature space analysis if available
        if 'feature_analysis' in results:
            for method in ['pca', 'tsne']:
                coords = results['feature_analysis'][method]['coordinates']
                error_mask = results['feature_analysis'][method]['error_mask']

                plt.figure(figsize=(10, 10))
                plt.scatter(
                    coords[~error_mask, 0],
                    coords[~error_mask, 1],
                    c='blue',
                    alpha=0.5,
                    label='Correct'
                )
                plt.scatter(
                    coords[error_mask, 0],
                    coords[error_mask, 1],
                    c='red',
                    alpha=0.5,
                    label='Error'
                )
                plt.title(f'Feature Space Analysis ({method.upper()})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.output_dir / f'feature_space_{method}.png')
                plt.close()

                