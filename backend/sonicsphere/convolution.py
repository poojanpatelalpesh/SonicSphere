"""Linear convolution: direct reference and FFT-based implementations.

All functions compute the full linear convolution y = x * h of length
len(x) + len(h) - 1 unless documented otherwise.

* :func:`direct_convolve`   - time-domain sum  y[n] = sum_k h[k] x[n-k]  (O(N*L))
* :func:`fft_convolve`      - one zero-padded FFT  (O((N+L) log(N+L)))
* :func:`overlap_add`       - block FFT convolution, overlap-add
* :func:`overlap_save`      - block FFT convolution, overlap-save
* :class:`DirectBlockConvolver` / :class:`PartitionedConvolver`
                            - streaming block processors used by the real-time renderer
                              (the latter is uniformly-partitioned overlap-save, UPOLS)
"""

from __future__ import annotations

import numpy as np

from .hrir_preprocessing import next_pow2


def _check(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x and h must be 1-D")
    if h.size == 0:
        raise ValueError("filter must not be empty")
    return x, h


def output_length(n_x: int, n_h: int) -> int:
    return 0 if n_x == 0 else n_x + n_h - 1


def direct_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Direct linear convolution. numpy.convolve evaluates the sum in the time domain."""
    x, h = _check(x, h)
    if x.size == 0:
        return np.zeros(0)
    return np.convolve(x, h, mode="full")


def direct_convolve_reference(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Literal implementation of y[n] = sum_k h[k] x[n-k] (slow; used in tests only)."""
    x, h = _check(x, h)
    n = output_length(x.size, h.size)
    y = np.zeros(n)
    for k, hk in enumerate(h):
        y[k: k + x.size] += hk * x
    return y


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Single-block FFT convolution: zero-pad both to >= N+L-1 to avoid circular wrap."""
    x, h = _check(x, h)
    n = output_length(x.size, h.size)
    if n == 0:
        return np.zeros(0)
    nfft = next_pow2(n)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def overlap_add(x: np.ndarray, h: np.ndarray, block: int = 1024) -> np.ndarray:
    """Overlap-add: convolve blocks of B input samples with h using an FFT of size
    >= B + L - 1, add the overlapping L-1 sample tails."""
    x, h = _check(x, h)
    n = output_length(x.size, h.size)
    if n == 0:
        return np.zeros(0)
    nfft = next_pow2(block + h.size - 1)
    H = np.fft.rfft(h, nfft)
    y = np.zeros(n + nfft)
    for start in range(0, x.size, block):
        seg = x[start: start + block]
        y[start: start + nfft] += np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)
    return y[:n]


def overlap_save(x: np.ndarray, h: np.ndarray, block: int | None = None) -> np.ndarray:
    """Overlap-save: circularly convolve overlapping frames of size N = B + L - 1 and keep
    the last B samples of each frame (the first L-1 are corrupted by wrap-around)."""
    x, h = _check(x, h)
    n_out = output_length(x.size, h.size)
    if n_out == 0:
        return np.zeros(0)
    L = h.size
    nfft = next_pow2(2 * L) if block is None else next_pow2(block + L - 1)
    B = nfft - L + 1
    H = np.fft.rfft(h, nfft)
    padded = np.concatenate([np.zeros(L - 1), x, np.zeros(B + L)])
    y = np.zeros(((n_out + B - 1) // B) * B)
    for i, start in enumerate(range(0, n_out, B)):
        frame = padded[start: start + nfft]
        if frame.size < nfft:
            frame = np.pad(frame, (0, nfft - frame.size))
        y[i * B: (i + 1) * B] = np.fft.irfft(np.fft.rfft(frame) * H, nfft)[L - 1:]
    return y[:n_out]


class DirectBlockConvolver:
    """Streaming direct convolution with a persistent input history.

    The filter can be swapped between blocks without disturbing the input history, so
    the output of the old and new filters for the same block can be crossfaded.
    """

    def __init__(self, filter_length: int, block_size: int):
        self.L = filter_length
        self.B = block_size
        self.history = np.zeros(self.L - 1 + self.B)

    def push(self, block: np.ndarray) -> None:
        self.history = np.concatenate([self.history[self.B:], block])

    def apply(self, h: np.ndarray) -> np.ndarray:
        """Output block for the most recently pushed input block, with filter h."""
        return np.convolve(self.history, h, mode="valid")


class PartitionedConvolver:
    """Uniformly-partitioned overlap-save (UPOLS) streaming convolution.

    Block size B, FFT size 2B, filter split into P = L / B partitions. A frequency-domain
    delay line (FDL) keeps the spectra of the last P input frames:

        Y_t[k] = sum_p X_{t-p}[k] * H_p[k],   y_t = last B samples of IFFT(Y_t)

    The latency equals the block size B (no extra algorithmic delay when B equals the
    host block size). Mirrors frontend/src/dsp/convolver.ts.
    """

    def __init__(self, filter_length: int, block_size: int):
        if filter_length % block_size:
            raise ValueError("filter length must be a multiple of the block size")
        self.B = block_size
        self.P = filter_length // block_size
        self.N = 2 * block_size
        self.prev = np.zeros(block_size)
        self.fdl = np.zeros((self.P, self.B + 1), dtype=complex)

    def partition(self, h: np.ndarray) -> np.ndarray:
        """Partition spectra H_p (P, B+1) of a filter of length P*B."""
        parts = np.asarray(h, dtype=float).reshape(self.P, self.B)
        return np.fft.rfft(parts, n=self.N, axis=-1)

    def push(self, block: np.ndarray) -> None:
        frame = np.concatenate([self.prev, block])
        self.prev = np.asarray(block, dtype=float).copy()
        self.fdl = np.roll(self.fdl, 1, axis=0)
        self.fdl[0] = np.fft.rfft(frame)

    def apply(self, H: np.ndarray) -> np.ndarray:
        Y = np.einsum("pk,pk->k", self.fdl, H)
        return np.fft.irfft(Y, self.N)[self.B:]


def partitioned_convolve(x: np.ndarray, h: np.ndarray, block: int = 128) -> np.ndarray:
    """Full linear convolution computed block-by-block with :class:`PartitionedConvolver`."""
    x, h = _check(x, h)
    n_out = output_length(x.size, h.size)
    if n_out == 0:
        return np.zeros(0)
    L = int(np.ceil(h.size / block) * block)
    hp = np.zeros(L)
    hp[: h.size] = h
    conv = PartitionedConvolver(L, block)
    H = conv.partition(hp)
    n_blocks = (n_out + block - 1) // block
    xp = np.zeros(n_blocks * block)
    xp[: x.size] = x
    out = np.empty(n_blocks * block)
    for b in range(n_blocks):
        conv.push(xp[b * block: (b + 1) * block])
        out[b * block: (b + 1) * block] = conv.apply(H)
    return out[:n_out]


def binaural_convolve(x: np.ndarray, hrir: np.ndarray, method: str = "direct", block: int = 128) -> np.ndarray:
    """y_L = x * h_L, y_R = x * h_R  ->  array (len(x)+L-1, 2), column 0 = left."""
    hrir = np.asarray(hrir, dtype=float)
    if hrir.shape[0] != 2:
        raise ValueError("hrir must have shape (2, L) with row 0 = left ear")
    fn = {
        "direct": direct_convolve,
        "fft": fft_convolve,
        "overlap_add": lambda a, b: overlap_add(a, b, 1024),
        "overlap_save": lambda a, b: overlap_save(a, b),
        "partitioned": lambda a, b: partitioned_convolve(a, b, block),
    }.get(method)
    if fn is None:
        raise ValueError(f"Unknown convolution method '{method}'")
    return np.stack([fn(x, hrir[0]), fn(x, hrir[1])], axis=1)
