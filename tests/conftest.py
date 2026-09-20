"""CPU uses NumPy; opt in to actual CUDA explicitly."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['XP_RUNTIME'] = ('CPU' if os.environ.get('TORCHLESS_TEST_DEVICE', 'cpu') == 'cpu'
                            else 'CUDA')
