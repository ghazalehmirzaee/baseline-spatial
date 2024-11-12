# test_imports.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python path: {sys.path}")

    try:
        from src.data.datasets import ChestXrayDataset
        print("✓ Successfully imported ChestXrayDataset")

        from src.models.integration import IntegratedModel
        print("✓ Successfully imported IntegratedModel")

        from src.utils.metrics import MetricTracker
        print("✓ Successfully imported MetricTracker")

        print("\nAll imports successful!")
        return True
    except Exception as e:
        print(f"\nError during imports: {e}")
        print(f"Error type: {type(e)}")
        print("\nTroubleshooting required!")
        return False


if __name__ == "__main__":
    test_imports()
    