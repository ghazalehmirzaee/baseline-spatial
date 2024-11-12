#!/bin/bash

# Clean up old installation
pip uninstall -y baseline_spatial
rm -rf build dist *.egg-info

# Install in development mode
pip install -e .

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Verify installation
python setup.py

