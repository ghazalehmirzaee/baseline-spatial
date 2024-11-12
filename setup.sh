#!/bin/bash

# Navigate to project directory
cd /users/gm00051/projects/cvpr/baseline-spatial

# Clean up old files
rm -rf baseline_spatial/__pycache__ baseline_spatial/**/__pycache__
rm -rf build dist *.egg-info

# Create directory structure
mkdir -p baseline_spatial/data
mkdir -p baseline_spatial/models/graph
mkdir -p baseline_spatial/utils
mkdir -p scripts
mkdir -p configs

# Create __init__.py files
echo 'from .data.loaders import create_data_loaders
from .data.datasets import ChestXrayDataset' > baseline_spatial/__init__.py

echo 'from .loaders import create_data_loaders
from .datasets import ChestXrayDataset' > baseline_spatial/data/__init__.py

echo 'from .integration import IntegratedModel' > baseline_spatial/models/__init__.py

echo 'from .metrics import MetricTracker
from .checkpointing import CheckpointManager' > baseline_spatial/utils/__init__.py

touch baseline_spatial/models/graph/__init__.py

# Create setup.py
cat > setup.py << 'EOL'
from setuptools import setup, find_packages

setup(
    name="baseline_spatial",
    version="0.1.0",
    packages=find_packages(),
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

# Set permissions
chmod 644 setup.py
chmod -R 644 baseline_spatial
find baseline_spatial -type d -exec chmod 755 {} \;

# Install package
pip uninstall -y baseline_spatial
pip install -e .

# Set PYTHONPATH
export PYTHONPATH="/users/gm00051/projects/cvpr/baseline-spatial:${PYTHONPATH}"

# Verify setup
echo "Verifying setup..."
ls -R baseline_spatial/
python -c "import baseline_spatial; print('Package imported successfully')"

