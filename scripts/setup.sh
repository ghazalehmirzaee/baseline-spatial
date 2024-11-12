#!/bin/bash
# scripts/setup.sh

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Set up pre-commit hooks
pip install pre-commit
pre-commit install

# Create necessary directories
mkdir -p checkpoints logs outputs

# Initialize wandb
wandb login

echo "Setup completed successfully!"

