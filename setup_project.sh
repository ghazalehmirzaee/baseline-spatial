#!/bin/bash

# Create directory structure
mkdir -p src/data src/models/graph src/utils configs scripts

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/models/graph/__init__.py

# Make sure the src directory is a package
echo "from .data.loaders import create_data_loaders" > src/__init__.py
echo "from .data.datasets import ChestXrayDataset" >> src/__init__.py

# Update data/__init__.py
cat > src/data/__init__.py << EOL
from .loaders import create_data_loaders, custom_collate_fn
from .datasets import ChestXrayDataset

__all__ = [
    'create_data_loaders',
    'custom_collate_fn',
    'ChestXrayDataset',
]
EOL

# Re-install the package
pip uninstall -y baseline-spatial
pip install -e .

# Set the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/users/gm00051/projects/cvpr/baseline-spatial"

