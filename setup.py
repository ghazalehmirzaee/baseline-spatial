from setuptools import setup, find_packages

setup(
    name="baseline_spatial",
    version="0.1",
    packages=find_packages(),
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
