import os
import sys
from pathlib import Path


def test_imports():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")

    try:
        print("\nTesting individual imports:")
        print("-" * 50)

        # Test src.data imports
        print("Testing src.data imports...")
        from src.data.datasets import ChestXrayDataset
        print("✓ ChestXrayDataset")

        # Test src.models imports
        print("\nTesting src.models imports...")
        from src.models.integration import IntegratedModel
        print("✓ IntegratedModel")

        # Test src.utils imports
        print("\nTesting src.utils imports...")
        from src.utils.metrics import MetricTracker
        from src.utils.checkpointing import CheckpointManager
        from src.utils.optimization import CosineAnnealingWarmupRestarts
        print("✓ MetricTracker")
        print("✓ CheckpointManager")
        print("✓ CosineAnnealingWarmupRestarts")

        print("\n✓ All imports successful!")
        return True

    except Exception as e:
        print(f"\n✗ Error during imports: {str(e)}")
        print(f"Error type: {type(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Add project root to Python path
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("\nTesting imports...")
    print(f"Project root: {PROJECT_ROOT}")
    test_imports()

