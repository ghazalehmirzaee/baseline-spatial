import os
import sys
from pathlib import Path


def verify_setup():
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    print("\nChecking files and directories:")
    paths = [
        'src',
        'src/data',
        'src/data/__init__.py',
        'src/data/loaders.py',
        'setup.py',
        '__init__.py'
    ]

    for path in paths:
        full_path = project_root / path
        exists = full_path.exists()
        print(f"{'✓' if exists else '✗'} {path} {'exists' if exists else 'missing'}")
        if exists and path.endswith('.py'):
            with open(full_path, 'r') as f:
                content = f.read()
                print(f"  File size: {len(content)} bytes")

    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    print("\nTrying imports...")
    try:
        from src.data.loaders import create_data_loaders
        print("✓ Successfully imported create_data_loaders")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_setup()

