import numpy as np
import pytest

from sonicsphere.convolution import (DirectBlockConvolver, PartitionedConvolver, binaural_convolve,
                                     direct_convolve, direct_convolve_reference, fft_convolve, output_length,
                                     overlap_add, overlap_save, partitioned_convolve)

FFT_METHODS = [fft_convolve, lambda x, h: overlap_add(x, h, 64), lambda x, h: overlap_add(x, h, 1024),
               overlap_save, lambda x, h: overlap_save(x, h, 100), lambda x, h: partitioned_convolve(x, h, 128),
               lambda x, h: partitioned_convolve(x, h, 32)]


@pytest.mark.parametrize("n,l", [(1, 1), (1, 200), (5, 200), (127, 200), (128, 256), (1000, 200), (4097, 256)])
def test_direct_matches_literal_definition(n, l, rng):
    x, h = rng.standard_normal(n), rng.standard_normal(l)
    np.testing.assert_allclose(direct_convolve(x, h), direct_convolve_reference(x, h), atol=1e-12)


@pytest.mark.parametrize("fn", FFT_METHODS)
@pytest.mark.parametrize("n,l", [(1, 200), (3, 200), (200, 200), (1000, 200), (5000, 256), (12345, 199)])
def test_fft_methods_equal_direct(fn, n, l, rng):
    x, h = rng.standard_normal(n), rng.standard_normal(l)
    ref = direct_convolve(x, h)
    y = fn(x, h)
    assert y.shape == ref.shape == (output_length(n, l),)
    assert np.max(np.abs(y - ref)) <= 1e-10 * max(1.0, np.max(np.abs(ref)))


def test_output_length():
    assert output_length(100, 200) == 299
    assert output_length(0, 200) == 0
    assert direct_convolve(np.zeros(0), np.ones(4)).size == 0
    assert fft_convolve(np.zeros(0), np.ones(4)).size == 0


def test_impulse_returns_filter(rng):
    h = rng.standard_normal(200)
    x = np.zeros(50)
    x[7] = 1.0
    for fn in [direct_convolve] + FFT_METHODS:
        y = fn(x, h)
        np.testing.assert_allclose(y[7:207], h, atol=1e-10)
        assert np.max(np.abs(y[:7])) < 1e-10 and np.max(np.abs(y[207:])) < 1e-10


def test_silence_gives_silence(rng):
    h = rng.standard_normal(200)
    for fn in [direct_convolve] + FFT_METHODS:
        assert np.max(np.abs(fn(np.zeros(300), h))) < 1e-12


def test_binaural_left_right_assignment(rng):
    x = rng.standard_normal(400)
    hl = np.zeros(16); hl[0] = 1.0           # left = identity
    hr = np.zeros(16); hr[3] = -0.5          # right = delayed, inverted, attenuated
    for m in ("direct", "fft", "overlap_add", "overlap_save", "partitioned"):
        y = binaural_convolve(x, np.stack([hl, hr]), m)
        np.testing.assert_allclose(y[:400, 0], x, atol=1e-10)
        np.testing.assert_allclose(y[3:403, 1], -0.5 * x, atol=1e-10)


def test_binaural_rejects_bad_shapes():
    with pytest.raises(ValueError):
        binaural_convolve(np.ones(10), np.ones((3, 8)))
    with pytest.raises(ValueError):
        binaural_convolve(np.ones(10), np.ones((2, 8)), "nope")


def test_streaming_block_convolvers_match_offline(rng):
    B, L = 128, 256
    h = rng.standard_normal(L)
    x = rng.standard_normal(B * 20)
    ref = direct_convolve(x, h)[: x.size]
    d, p = DirectBlockConvolver(L, B), PartitionedConvolver(L, B)
    H = p.partition(h)
    yd, yp = [], []
    for b in range(20):
        blk = x[b * B:(b + 1) * B]
        d.push(blk); p.push(blk)
        yd.append(d.apply(h)); yp.append(p.apply(H))
    np.testing.assert_allclose(np.concatenate(yd), ref, atol=1e-10)
    np.testing.assert_allclose(np.concatenate(yp), ref, atol=1e-10)


def test_partitioned_requires_block_multiple():
    with pytest.raises(ValueError):
        PartitionedConvolver(200, 128)
