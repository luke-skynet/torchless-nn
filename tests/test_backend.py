"""Backend selection must work independently of installed CUDA packages."""
import os
from pathlib import Path
import subprocess
import sys


def test_numpy_scatter_ignores_cupyx():
    subprocess.run([sys.executable, '-c', '''
import sys
import types
import numpy as np

cupyx = types.ModuleType('cupyx')
def fail(*args):
    raise AssertionError('CPU scatter must not call cupyx')
cupyx.scatter_add = fail
sys.modules['cupyx'] = cupyx
from backend import xp, to_numpy
from transformer_adapters import GPTEmbeddingTable, GPTEmbedFront
assert xp is np
target = xp.zeros((3, 2), dtype=xp.float32)
layer = GPTEmbedFront(GPTEmbeddingTable(3, 2, table=target), context_length=3, positional='none')
layer.forward(xp.array([[1, 1, 2]]))
layer.backward(xp.array([[[3, 6], [9, 12], [15, 18]]], dtype=xp.float32))
np.testing.assert_array_equal(to_numpy(layer.table_grads), [[0, 0], [12, 18], [15, 18]])
assert 'cupy' not in sys.modules
'''], cwd=Path(__file__).resolve().parents[1],
                   env={**os.environ, 'TORCHLESS_BACKEND': 'numpy'}, check=True)
