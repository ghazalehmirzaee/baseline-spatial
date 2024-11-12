import sys
from pathlib import Path
import importlib.util


def check_module(module_name, file_path):
    print(f"\nChecking {module_name}...")
    print(f"Looking for file: {file_path}")
    print(f"File exists: {Path(file_path).exists()}")

    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"Module can be imported from: {spec.origin}")
    else:
        print(f"Module not found in Python path")


project_root = Path(__file__).resolve().parent
print(f"Project root: {project_root}")
print("\nPython path:")
for p in sys.path:
    print(f"  {p}")

check_module('src.data.loaders', project_root / 'src' / 'data' / 'loaders.py')
check_module('src.data.datasets', project_root / 'src' / 'data' / 'datasets.py')


