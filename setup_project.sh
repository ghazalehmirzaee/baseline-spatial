#!/bin/bash

cd /users/gm00051/projects/cvpr/baseline-spatial

# Clean up cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -r {} +
rm -rf *.egg-info build dist

# Create proper structure (if directories don't exist)
mkdir -p src/data
mkdir -p src/models/graph
mkdir -p src/utils
mkdir -p scripts
mkdir -p configs

# Create __init__.py files
cat > __init__.py << 'EOL'
from src.data.loaders import create_data_loaders
from src.data.datasets import ChestXrayDataset
EOL

cat > src/__init__.py << 'EOL'
from .data.loaders import create_data_loaders
from .data.datasets import ChestXrayDataset
EOL

cat > src/data/__init__.py << 'EOL'
from .loaders import create_data_loaders, custom_collate_fn
from .datasets import ChestXrayDataset

__all__ = [
    'create_data_loaders',
    'custom_collate_fn',
    'ChestXrayDataset'
]
EOL

# Create setup.py
cat > setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="baseline-spatial",
    version="0.1.0",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "Pillow>=8.0.0",
        "wandb>=0.12.0",
        "tqdm>=4.50.0",
        "PyYAML>=5.4.0",
    ],
)
EOL

# Set correct permissions
chmod 644 setup.py
chmod 644 __init__.py
chmod 644 src/__init__.py
chmod 644 src/data/__init__.py

# Create/update loaders.py
cat > src/data/loaders.py << 'EOL'
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import os
import torch.distributed as dist

def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function to handle batches with potential None values."""
    images, labels, bboxes = zip(*batch)

    # Stack images and labels
    images = torch.stack(images)
    labels = torch.stack(labels)

    # Handle bounding boxes
    if all(bbox is not None for bbox in bboxes):
        bboxes = torch.stack(bboxes)
    else:
        # Create zero tensor if any bbox is None
        first_valid_bbox = next((bbox for bbox in bboxes if bbox is not None), None)
        if first_valid_bbox is not None:
            bbox_shape = first_valid_bbox.shape
            bboxes = torch.zeros(len(batch), *bbox_shape, device=first_valid_bbox.device)
            for i, bbox in enumerate(bboxes):
                if bbox is not None:
                    bboxes[i] = bbox
        else:
            bboxes = torch.zeros(len(batch), 14, 4)

    return images, labels, bboxes

def create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size: int,
        num_workers: int,
        distributed: bool = False
) -> Dict[str, DataLoader]:
    """Create data loaders with custom collate function."""

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
EOL

chmod 644 src/data/loaders.py

