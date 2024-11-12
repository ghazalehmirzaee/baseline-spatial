import os
import sys
from pathlib import Path


def verify_setup():
    project_root = Path(__file__).resolve().parent

    print(f"Project root: {project_root}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    print("\nChecking directory structure:")
    paths = [
        'baseline_spatial',
        'baseline_spatial/data',
        'baseline_spatial/models',
        'baseline_spatial/utils',
        'scripts',
        'configs'
    ]

    for path in paths:
        full_path = project_root / path
        exists = full_path.exists()
        print(f"{'✓' if exists else '✗'} {path} {'exists' if exists else 'missing'}")


if __name__ == "__main__":
    verify_setup()

