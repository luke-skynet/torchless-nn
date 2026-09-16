import numpy as np
import pytest
import torch
from backend import xp

from activations import GeLU, GeLUTanh


def host(value):
    return value.get() if hasattr(value, 'get') else np.asarray(value)


@pytest.mark.parametrize('activation,approximate', [(GeLU, 'none'), (GeLUTanh, 'tanh')])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_gelu_matches_pytorch(activation, approximate, dtype):
    # Include both tails, zero, and the region where the two formulas differ.
    values = np.array([-10., -4., -2.5, -1., -.1, 0., .1, 1., 2.5, 4., 10.], dtype=dtype)
    upstream = np.linspace(-1., 2., len(values), dtype=dtype)
    reference = torch.tensor(values, requires_grad=True)
    expected = torch.nn.functional.gelu(reference, approximate=approximate)
    expected.backward(torch.tensor(upstream))

    layer = activation()
    x, gradient = xp.asarray(values), xp.asarray(upstream)
    actual = layer.forward(x)
    dx = layer.backward(gradient)
    tolerance = 3e-7 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(host(actual), expected.detach().numpy(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(host(dx), reference.grad.numpy(), atol=tolerance, rtol=tolerance)
    assert actual.dtype == values.dtype and dx.dtype == values.dtype
    np.testing.assert_array_equal(host(x), values)
    np.testing.assert_array_equal(host(gradient), upstream)
    layer.clear_cache()
    assert all(getattr(layer, name) is None for name in layer.CACHED)
