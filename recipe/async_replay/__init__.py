"""GA-ARL recipe with repository-local configuration and utilities."""
import os
from pathlib import Path

os.environ.setdefault("GA_ARL_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
