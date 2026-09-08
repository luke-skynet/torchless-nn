"""CPU uses NumPy as the CuPy-compatible backend; opt in to actual CUDA explicitly."""
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get('TORCHLESS_TEST_DEVICE', 'cpu') == 'cpu':
    sys.modules['cupy'] = np
