"""Force mock paths for all tests so we don't need GPU or AWS."""
import os
os.environ.setdefault("TROPHIC_MOCK_MODELS", "1")
os.environ.setdefault("TROPHIC_MOCK_PREDATOR", "1")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
