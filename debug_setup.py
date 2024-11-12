import os
import sys
from pathlib import Path


def check_environment():
    print("=== Environment Check ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"\nPYTHONPATH:")
    for path in sys.path:
        print(f"  {path}")

    print("\n=== Project Structure ===")
    project_root = Path(os.getcwd())
    print(f"Project root: {project_root}")

    # Check critical files and directories
    critical_paths = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/loaders.py',
        'src/data/datasets.py',
        'setup.py',
        'scripts/train_integrated.py',
        'configs/integrated_model_config.yaml'
    ]

    for path in critical_paths:
        full_path = project_root / path
        status = '✓' if full_path.exists() else '✗'
        print(f"{status} {path} {'exists' if status == '✓' else 'missing'}")

    print("\n=== Import Test ===")
    try:
        from src.data.loaders import create_data_loaders
        print("✓ Successfully imported create_data_loaders")
    except ImportError as e:
        print(f"✗ Failed to import create_data_loaders: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_environment()

    