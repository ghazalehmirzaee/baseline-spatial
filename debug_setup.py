import os
import sys
from pathlib import Path


def verify_setup():
    base_path = Path(__file__).resolve().parent

    # Check directory structure
    paths_to_check = [
        'src',
        'src/data',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/loaders.py'
    ]

    print("Checking directory structure:")
    for path in paths_to_check:
        full_path = base_path / path
        exists = full_path.exists()
        print(f"{'✓' if exists else '✗'} {path} {'exists' if exists else 'missing'}")
        if exists and path.endswith('.py'):
            with open(full_path, 'r') as f:
                content = f.read()
                print(f"  File size: {len(content)} bytes")

    # Check Python path
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    # Try imports
    print("\nTesting imports:")
    try:
        import src
        print("✓ Successfully imported src")
    except ImportError as e:
        print(f"✗ Failed to import src: {e}")

    try:
        from src.data import loaders
        print("✓ Successfully imported src.data.loaders")
    except ImportError as e:
        print(f"✗ Failed to import src.data.loaders: {e}")


if __name__ == "__main__":
    verify_setup()

