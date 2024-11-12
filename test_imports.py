import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_imports():
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    print("\nTesting imports...")

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

    try:
        from src.data.loaders import create_data_loaders
        print("✓ Successfully imported create_data_loaders")
    except ImportError as e:
        print(f"✗ Failed to import create_data_loaders: {e}")


if __name__ == "__main__":
    test_imports()

