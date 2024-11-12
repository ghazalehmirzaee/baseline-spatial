import os
import sys
from pathlib import Path


def verify_setup():
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    # Check Python path
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    # Check directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(project_root / 'baseline_spatial'):
        level = root.replace(str(project_root), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    # Try imports
    print("\nTesting imports...")
    try:
        import baseline_spatial
        print("✓ Successfully imported baseline_spatial")

        from baseline_spatial.data import create_data_loaders
        print("✓ Successfully imported create_data_loaders")
    except ImportError as e:
        print(f"✗ Import error: {e}")


if __name__ == "__main__":
    verify_setup()

