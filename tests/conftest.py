import sys
from pathlib import Path

# Tests in sub-directories share helpers that live next to the top-level tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
