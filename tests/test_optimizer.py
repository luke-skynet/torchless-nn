"""Adam parity and storage safety; run with TORCHLESS_TEST_DEVICE=cuda for the kernel."""
import numpy as np
import pytest

from backend import xp, to_numpy
from network import Network
from optimizer import Adam
from transformer_adapters import VitProjector
from utils import Layer
import backend


def make_entry(shape, dtype, decay=True):
    rng = np.random.default_rng(23)
    param = xp.asarray(rng.normal(size=shape), dtype=dtype)
    grad = xp.zeros_like(param)
    moment = xp.asarray(rng.normal(scale=.1, size=shape), dtype=dtype)
    variance = xp.asarray(rng.uniform(.01, .1, size=shape), dtype=dtype)
    return param, grad, moment, variance, decay


def reference_step(entry, gradient, t, samples, eps):
    # Independent NumPy Adam equations, including nonzero initial optimizer state.
    p, _, m, v, decay = entry
    g = gradient / samples
    m[...] = .9 * m + (1 - .9) * g
    v[...] = .999 * v + (1 - .999) * g**2
    p[...] -= .003 * ((m / (1 - .9**t)) / (np.sqrt(v / (1 - .999**t)) + eps)
                     + (.02 if decay else 0) * p)


@pytest.mark.parametrize('eps', [1e-7, .05])
def test_multistep_mixed_dtype_and_chunk_boundaries(eps):
    # Both dtypes, scalar and empty tensors, and lengths either side of a chunk.
    entries = [make_entry(shape, dtype, decay)
               for shape, dtype, decay in [((3, 7), xp.float32, True),
                                           ((4095,), xp.float64, False),
                                           ((4096,), xp.float32, False),
                                           ((4103,), xp.float64, True),
                                           ((), xp.float32, True),
                                           ((0,), xp.float32, False)]]
    expected = [tuple(to_numpy(a).copy() for a in e[:4]) + (e[4],) for e in entries]
    optimizer = Adam()
    rng = np.random.default_rng(44)
    for t, samples in enumerate([7, 3, 11, 1], start=1):
        for entry, ref in zip(entries, expected):
            gradient = np.asarray(rng.normal(size=entry[0].shape), dtype=entry[0].dtype)
            entry[1][...] = xp.asarray(gradient)
            reference_step(ref, gradient, t, samples, eps)
        optimizer.step(entries, .003, .02, t, samples, eps)
        for entry, ref in zip(entries, expected):
            tolerance = 2e-6 if entry[0].dtype == xp.float32 else 2e-13
            for index in (0, 2, 3):
                np.testing.assert_allclose(to_numpy(entry[index]), ref[index],
                                           rtol=tolerance, atol=tolerance)
            assert not bool(xp.any(entry[1]))


def test_shared_storage_updated_once_and_metadata_rebuilt():
    entry = make_entry((2, 3), xp.float64)
    expected = tuple(to_numpy(a).copy() for a in entry[:4]) + (True,)
    entry[1].fill(2)
    reference_step(expected, np.full((2, 3), 2.), 1, 2, 1e-7)
    optimizer = Adam()
    # Distinct array objects referencing exactly the same storage count once.
    alias = tuple(a.view() for a in entry[:4]) + (True,)
    optimizer.step([entry, alias], .003, .02, 1, 2)
    np.testing.assert_allclose(to_numpy(entry[0]), expected[0], rtol=1e-13)
    assert len(optimizer.entries) == 1
    signature = optimizer._signature
    metadata = getattr(optimizer, '_tensors', None)
    optimizer.step([entry, alias], .003, .02, 2, 2)
    assert optimizer._signature is signature
    if xp is not np:
        assert optimizer._tensors is metadata

    old = [a.copy() for a in entry[:4]]
    replacement = tuple(a.copy() for a in entry[:4]) + (False,)
    replacement[1].fill(3)
    expected = tuple(to_numpy(a).copy() for a in replacement[:4]) + (False,)
    reference_step(expected, np.full((2, 3), 3.), 3, 2, 1e-7)
    optimizer.step([replacement], .003, .02, 3, 2)
    assert optimizer._signature != signature
    np.testing.assert_allclose(to_numpy(replacement[0]), expected[0], rtol=1e-13)
    for original, saved in zip(entry[:4], old):
        np.testing.assert_array_equal(to_numpy(original), to_numpy(saved))


def test_decay_policy_change_rebuilds_metadata():
    entry = make_entry((2, 3), xp.float64)
    optimizer = Adam()
    optimizer.step([entry], .003, .02, 1, 2)
    signature = optimizer._signature
    changed = entry[:4] + (False,)
    expected = tuple(to_numpy(a).copy() for a in entry[:4]) + (False,)
    reference_step(expected, np.zeros((2, 3)), 2, 2, 1e-7)
    optimizer.step([changed], .003, .02, 2, 2)
    assert optimizer._signature != signature
    np.testing.assert_allclose(to_numpy(entry[0]), expected[0], rtol=1e-13)


@pytest.mark.parametrize('conflict', ['state', 'decay', 'overlap'])
def test_rejects_unsafe_shared_storage_before_mutation(conflict):
    entry = make_entry((2, 3), xp.float32)
    other = list(entry)
    if conflict == 'state':
        other[1] = other[1].copy()
    elif conflict == 'decay':
        other[4] = False
    else:
        other[:4] = [a[1:] for a in other[:4]]
    saved = entry[0].copy()
    with pytest.raises(ValueError, match='Shared parameters|overlap'):
        Adam().step([entry, tuple(other)], .003, .02, 1, 2)
    np.testing.assert_array_equal(to_numpy(entry[0]), to_numpy(saved))


def test_rejects_mismatched_state():
    entry = list(make_entry((2, 3), xp.float32))
    entry[1] = xp.zeros((3, 2), dtype=xp.float32)
    with pytest.raises(ValueError, match='shape and dtype'):
        Adam().step([entry], .003, .02, 1, 2)


def test_empty_optimizer():
    Adam().step([], .003, .02, 1, 2)


@pytest.mark.parametrize('cls_token', [False, True])
def test_network_decay_exclusions_and_frozen_positions(monkeypatch, cls_token):
    monkeypatch.setattr(backend, 'FLOAT_TYPE', xp.float64)
    dense = Layer()
    for shape in [(2, 3), (3,)]:
        dense.register(xp.ones(shape, dtype=xp.float64))
    vit = VitProjector((2, 4, 6), (2, 3), 4, num_registers=2, cls_token=cls_token)
    model = Network([dense, vit])
    register_rows = slice(int(cls_token), int(cls_token) + 2)
    for step in range(1, 4):
        # Two backwards must accumulate before the update clears gradients.
        for _ in range(2):
            output = vit.forward(xp.ones((1, 2, 4, 6)))
            vit.backward(xp.ones_like(output))
        assert not bool(xp.any(vit.positional_encoding_grads[register_rows]))
        model._update(.003, .02, step, 2)
        assert not bool(xp.any(vit.positional_encodings[register_rows]))
        for layer in model.layers:
            assert all(not bool(xp.any(g)) for g in layer.gradients)
        assert not bool(xp.any(vit.moments[-1][register_rows]))
        assert not bool(xp.any(vit.variances[-1][register_rows]))
    np.testing.assert_allclose(to_numpy(dense.parameters[0]), (1 - .003 * .02)**3)
    np.testing.assert_array_equal(to_numpy(dense.parameters[1]), np.ones(3))
    assert [e[4] for e in model.optimizer.entries] == [True, False] + [False] * len(vit.parameters)


@pytest.mark.skipif(xp is np, reason='requires CUDA')
def test_cuda_one_launch_and_no_metadata_reupload(monkeypatch):
    entries = [make_entry((4103,), xp.float32), make_entry((17,), xp.float64)]
    optimizer = Adam()
    optimizer.step(entries, .003, .02, 1, 2)
    kernel = optimizer._kernel
    calls = []

    def launch(*args):
        calls.append(args)
        return kernel(*args)

    def unexpected_upload(*args, **kwargs):
        raise AssertionError('Unchanged optimizer storage must not upload metadata again')

    monkeypatch.setattr(optimizer, '_kernel', launch)
    monkeypatch.setattr(xp, 'asarray', unexpected_upload)
    optimizer.step(entries, .003, .02, 2, 2)
    xp.cuda.get_current_stream().synchronize()
    assert len(calls) == 1
    assert calls[0][0] == (3,)


@pytest.mark.skipif(xp is np, reason='requires CUDA')
def test_cuda_rejects_noncontiguous_storage():
    entry = make_entry((2, 3), xp.float32)
    transposed = tuple(a.T for a in entry[:4]) + (True,)
    with pytest.raises(ValueError, match='C-contiguous'):
        Adam().step([transposed], .003, .02, 1, 2)
