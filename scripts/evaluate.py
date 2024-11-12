# scripts/evaluate.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm

from src.models.integration import GraphAugmentedViT
from src.data.datasets import ChestXrayDataset
from src.visualization.attention import AttentionVisualizer
from src.visualization.gradcam import GradCAMVisualizer


class Evaluator:
    """Comprehensive evaluation of the model."""

    def __init__(
            self,
            model: nn.Module,
            device: str,
            output_dir: str,
            disease_names: List[str]
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.disease_names = disease_names

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'attention_maps').mkdir(exist_ok=True)
        (self.output_dir / 'gradcam').mkdir(exist_ok=True)

        # Initialize visualizers
        self.attention_vis = AttentionVisualizer()
        self.gradcam_vis = GradCAMVisualizer(model)

    @torch.no_grad()
    def evaluate(
            self,
            test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_features = []
        attention_maps = []

        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels, bb_coords = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            if bb_coords is not None:
                bb_coords = bb_coords.to(self.device)

            # Forward pass
            outputs = self.model(images, bb_coords)
            preds = torch.sigmoid(outputs['logits'])

            # Store results
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_features.append(outputs['vit_features'][:, 0].cpu())
            attention_maps.append(outputs['attention_weights'])

        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_features = torch.cat(all_features, dim=0).numpy()

        # Compute metrics
        metrics = self._compute_detailed_metrics(all_preds, all_labels)

        # Generate visualizations
        self._plot_roc_curves(all_preds, all_labels)
        self._plot_pr_curves(all_preds, all_labels)
        self._plot_confusion_matrices(all_preds, all_labels)
        self._visualize_feature_space(all_features, all_labels)

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _compute_detailed_metrics(
            self,
            preds: np.ndarray,
            labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}

        # Compute per-class metrics
        for i, disease in enumerate(self.disease_names):
            # AUC-ROC
            fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
            metrics[f"{disease}_auc"] = auc(fpr, tpr)

            # Average Precision
            precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
            metrics[f"{disease}_ap"] = auc(recall, precision)

            # F1 Score at optimal threshold
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            metrics[f"{disease}_f1"] = f1_scores[optimal_idx]

            # Sensitivity and Specificity at optimal threshold
            binary_preds = (preds[:, i] > 0.5).astype(int)
            tn = np.sum((labels[:, i] == 0) & (binary_preds == 0))
            fp = np.sum((labels[:, i] == 0) & (binary_preds == 1))
            fn = np.sum((labels[:, i] == 1) & (binary_preds == 0))
            tp = np.sum((labels[:, i] == 1) & (binary_preds == 1))

            metrics[f"{disease}_sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"{disease}_specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f"{disease}_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Compute mean metrics
        metrics["mean_auc"] = np.mean([metrics[k] for k in metrics if k.endswith("_auc")])
        metrics["mean_ap"] = np.mean([metrics[k] for k in metrics if k.endswith("_ap")])
        metrics["mean_f1"] = np.mean([metrics[k] for k in metrics if k.endswith("_f1")])
        metrics["mean_sensitivity"] = np.mean([metrics[k] for k in metrics if k.endswith("_sensitivity")])
        metrics["mean_specificity"] = np.mean([metrics[k] for k in metrics if k.endswith("_specificity")])
        metrics["mean_precision"] = np.mean([metrics[k] for k in metrics if k.endswith("_precision")])

        # Compute exact match ratio
        correct_predictions = np.all(((preds > 0.5) == labels).astype(int), axis=1)
        metrics["exact_match"] = np.mean(correct_predictions)

        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(preds, labels)
        metrics["confidence_intervals"] = confidence_intervals

        # Error analysis
        error_analysis = self._perform_error_analysis(preds, labels)
        metrics["error_analysis"] = error_analysis

        return metrics

    def _compute_confidence_intervals(
            self,
            preds: np.ndarray,
            labels: np.ndarray,
            n_bootstrap: int = 1000,
            confidence_level: float = 0.95
    ) -> Dict[str, List[float]]:
        """Compute confidence intervals using bootstrapping."""
        ci_metrics = {}
        n_samples = len(labels)
        alpha = (1 - confidence_level) / 2

        for i, disease in enumerate(self.disease_names):
            metric_distributions = {
                "auc": [],
                "ap": [],
                "f1": [],
                "sensitivity": [],
                "specificity": [],
                "precision": []
            }

            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_preds = preds[indices, i]
                boot_labels = labels[indices, i]

                # Compute metrics
                fpr, tpr, _ = roc_curve(boot_labels, boot_preds)
                metric_distributions["auc"].append(auc(fpr, tpr))

                precision, recall, _ = precision_recall_curve(boot_labels, boot_preds)
                metric_distributions["ap"].append(auc(recall, precision))

                binary_preds = (boot_preds > 0.5).astype(int)
                tn = np.sum((boot_labels == 0) & (binary_preds == 0))
                fp = np.sum((boot_labels == 0) & (binary_preds == 1))
                fn = np.sum((boot_labels == 1) & (binary_preds == 0))
                tp = np.sum((boot_labels == 1) & (binary_preds == 1))

                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                metric_distributions["f1"].append(f1)

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                metric_distributions["sensitivity"].append(sensitivity)

                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metric_distributions["specificity"].append(specificity)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                metric_distributions["precision"].append(precision)

            # Compute confidence intervals
            for metric in metric_distributions:
                values = np.array(metric_distributions[metric])
                ci_metrics[f"{disease}_{metric}_ci"] = [
                    np.percentile(values, alpha * 100),
                    np.percentile(values, (1 - alpha) * 100)
                ]

        # Compute mean confidence intervals
        for metric in ["auc", "ap", "f1", "sensitivity", "specificity", "precision"]:
            mean_values = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_preds = preds[indices]
                boot_labels = labels[indices]

                disease_metrics = []
                for i in range(len(self.disease_names)):
                    if metric == "auc":
                        fpr, tpr, _ = roc_curve(boot_labels[:, i], boot_preds[:, i])
                        disease_metrics.append(auc(fpr, tpr))
                    elif metric == "ap":
                        precision, recall, _ = precision_recall_curve(
                            boot_labels[:, i],
                            boot_preds[:, i]
                        )
                        disease_metrics.append(auc(recall, precision))
                    else:
                        binary_preds = (boot_preds[:, i] > 0.5).astype(int)
                        tn = np.sum((boot_labels[:, i] == 0) & (binary_preds == 0))
                        fp = np.sum((boot_labels[:, i] == 0) & (binary_preds == 1))
                        fn = np.sum((boot_labels[:, i] == 1) & (binary_preds == 0))
                        tp = np.sum((boot_labels[:, i] == 1) & (binary_preds == 1))

                        if metric == "f1":
                            val = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                        elif metric == "sensitivity":
                            val = tp / (tp + fn) if (tp + fn) > 0 else 0
                        elif metric == "specificity":
                            val = tn / (tn + fp) if (tn + fp) > 0 else 0
                        else:  # precision
                            val = tp / (tp + fp) if (tp + fp) > 0 else 0
                        disease_metrics.append(val)

                mean_values.append(np.mean(disease_metrics))

            ci_metrics[f"mean_{metric}_ci"] = [
                np.percentile(mean_values, alpha * 100),
                np.percentile(mean_values, (1 - alpha) * 100)
            ]

        return ci_metrics

    def _perform_error_analysis(
            self,
            preds: np.ndarray,
            labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        binary_preds = (preds > 0.5).astype(int)

        # Compute per-disease error rates
        error_rates = {}
        for i, disease in enumerate(self.disease_names):
            errors = (binary_preds[:, i] != labels[:, i]).mean()
            error_rates[disease] = errors

        # Compute error co-occurrence matrix
        n_diseases = len(self.disease_names)
        error_cooccurrence = np.zeros((n_diseases, n_diseases))

        for i in range(n_diseases):
            for j in range(n_diseases):
                errors_i = (binary_preds[:, i] != labels[:, i])
                errors_j = (binary_preds[:, j] != labels[:, j])
                error_cooccurrence[i, j] = np.mean(errors_i & errors_j)

        return {
            "per_disease_errors": {
                "Disease": self.disease_names,
                "Error Rate": list(error_rates.values())
            },
            "error_cooccurrence": error_cooccurrence.tolist()
        }

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics to JSON file."""
        output_file = self.output_dir / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    def _plot_roc_curves(
            self,
            preds: np.ndarray,
            labels: np.ndarray
    ) -> None:
        """Plot ROC curves for each disease."""
        plt.figure(figsize=(15, 10))

        for i, disease in enumerate(self.disease_names):
            fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plt.plot(
                fpr,
                tpr,
                label=f'{disease} (AUC = {roc_auc:.3f})'
            )

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'plots' / 'roc_curves.png')
        plt.close()

    def _plot_pr_curves(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Plot Precision-Recall curves for each disease."""
        plt.figure(figsize=(15, 10))

        for i, disease in enumerate(self.disease_names):
            precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
            pr_auc = auc(recall, precision)

            plt.plot(
                recall,
                precision,
                label=f'{disease} (AP = {pr_auc:.3f})'
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'plots' / 'pr_curves.png')
        plt.close()

    def _plot_confusion_matrices(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Plot confusion matrices for each disease."""
        binary_preds = (preds > 0.5).astype(int)

        for i, disease in enumerate(self.disease_names):
            cm = confusion_matrix(labels[:, i], binary_preds[:, i])
            plt.figure(figsize=(8, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )

            plt.title(f'Confusion Matrix - {disease}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.savefig(self.output_dir / 'plots' / f'confusion_matrix_{disease}.png')
            plt.close()

    def _visualize_feature_space(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        perplexity: int = 30
    ) -> None:
        """Visualize feature space using t-SNE."""
        from sklearn.manifold import TSNE

        # Reduce dimensionality
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)

        # Plot for each disease
        for i, disease in enumerate(self.disease_names):
            plt.figure(figsize=(10, 8))

            # Plot negative and positive samples
            neg_mask = labels[:, i] == 0
            pos_mask = labels[:, i] == 1

            plt.scatter(
                features_2d[neg_mask, 0],
                features_2d[neg_mask, 1],
                c='blue',
                label='Negative',
                alpha=0.5
            )
            plt.scatter(
                features_2d[pos_mask, 0],
                features_2d[pos_mask, 1],
                c='red',
                label='Positive',
                alpha=0.5
            )

            plt.title(f'Feature Space Visualization - {disease}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()

            plt.savefig(self.output_dir / 'plots' / f'tsne_{disease}.png')
            plt.close()

    def visualize_attention(
        self,
        image: torch.Tensor,
        bb_coords: Optional[torch.Tensor] = None
    ) -> None:
        """Visualize attention maps."""
        self.model.eval()

        # Get attention maps
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(
                image.unsqueeze(0).to(self.device),
                bb_coords.unsqueeze(0).to(self.device) if bb_coords is not None else None
            )

        # Save attention visualizations
        self.attention_vis.visualize(
            image.cpu(),
            attention_maps,
            save_dir=self.output_dir / 'attention_maps'
        )

    def visualize_gradcam(
        self,
        image: torch.Tensor,
        target_disease: int
    ) -> None:
        """Generate GradCAM visualization."""
        self.model.eval()

        # Generate GradCAM
        gradcam_map = self.gradcam_vis.generate(
            image.unsqueeze(0).to(self.device),
            target_disease
        )

        # Save visualization
        save_path = self.output_dir / 'gradcam' / f'gradcam_{self.disease_names[target_disease]}.png'
        self.gradcam_vis.save_visualization(image.cpu(), gradcam_map, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for evaluation')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create test dataset
    test_dataset = ChestXrayDataset(
        image_dir=config['data']['test_image_dir'],
        label_file=config['data']['test_label_file'],
        bbox_file=config['data']['bbox_file'],
        transform=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Create model and load checkpoint
    model = GraphAugmentedViT(**config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    # Initialize evaluator
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    evaluator = Evaluator(
        model=model,
        device=args.device,
        output_dir=config['paths']['output_dir'],
        disease_names=disease_names
    )

    # Run evaluation
    metrics = evaluator.evaluate(test_loader)
    print("\nEvaluation completed!")
    print(f"Mean AUC: {metrics['mean_auc']:.4f}")
    print(f"Mean AP: {metrics['mean_ap']:.4f}")

    # Generate visualizations for sample images
    sample_indices = [0, 100, 200, 300, 400]  # Modify as needed
    for idx in sample_indices:
        image, _, bb_coords = test_dataset[idx]
        evaluator.visualize_attention(image, bb_coords)
        for disease_idx in range(len(disease_names)):
            evaluator.visualize_gradcam(image, disease_idx)

if __name__ == '__main__':
    main()

