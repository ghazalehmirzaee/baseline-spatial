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

