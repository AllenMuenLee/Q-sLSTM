import sys
from pathlib import Path

# On some Windows installs, loading pandas' pyarrow DLLs after torch has initialized crashes the
# interpreter (access violation). Import pandas before any test module imports torch.
import pandas  # noqa: F401

# Tests in sub-directories share helpers that live next to the top-level tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fixtures shared by the solar-generation tests in several sub-directories.
pytest_plugins = ["solar_test_utils"]
