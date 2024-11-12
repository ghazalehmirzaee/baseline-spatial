# setup.py

from setuptools import setup, find_packages

setup(
    name="baseline-spatial",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'Pillow>=8.3.0',
        'pyyaml>=5.4.1',
        'wandb>=0.12.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.61.0'
    ]
)

