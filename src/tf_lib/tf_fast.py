"""
Faster drop-in versions of chirplet_ridge.chirplet_reassign (STFT, FSST, chirplet
transform, FCRM).

    chirplet_reassign_fast(x, backend="auto", ...)      dispatcher (same keywords and
                                                         return dict as chirplet_reassign)
    chirplet_reassign_parallel(x, n_jobs, backend)      multi-core CPU
    chirplet_reassign_gpu(x, xp="cupy", precision)      GPU (CuPy), or the same
                                                         vectorized code on NumPy

Why this works: every frame is independent. In the serial version ~10 % of the
time is FFTs and ~90 % element-wise complex arithmetic, indexing and scatter-adds
on one core.

Multi-core: the frames are split into blocks; each block is computed by the
unmodified chirplet_reassign on the signal segment it needs (padded by one window),
so the result equals the serial one, for every mode. backend="processes" (default)
writes results into shared memory; backend="threads" avoids process start-up but
is limited by the parts of NumPy that hold the GIL.

GPU: a vectorized port (modes "fsst", "eq9", "eq6", "joint"; not "two_step") that
processes large frame chunks, uses fixed-size arrays (out-of-range coefficients go
to a discarded bin instead of boolean compaction) and optionally single precision.
Results are copied to host memory chunk by chunk, so GPU memory stays bounded.
Requires CuPy (pip install cupy-cuda12x, matching your CUDA version).
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from . import chirplet_ridge as CR

TWO_PI = 2 * np.pi
_OUT_KEYS = {"R", "Rw_omega", "Rw_beta", "C2", "C", "omega_hat", "beta_hat"}


# =========================================================================
# Dispatcher
# =========================================================================
def gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def chirplet_reassign_fast(x, backend="auto", n_jobs=None, **kw):
    """Same keywords and return value as chirplet_ridge.chirplet_reassign.
    backend: "auto" (GPU if CuPy + a CUDA device, else processes if > 1 core, else
    serial), "serial", "processes", "threads", "gpu", "gpu-numpy" (GPU code on the CPU).
    precision: "single" | "double", used by the GPU backends only."""
    precision = kw.pop("precision", "single")      # GPU only
    if backend == "auto":
        if gpu_available() and kw.get("mode", "joint") != "two_step":
            backend = "gpu"
        elif (os.cpu_count() or 1) > 1:
            backend = "processes"
        else:
            backend = "serial"
    if backend == "serial":
        return CR.chirplet_reassign(x, **kw)
    if backend in ("processes", "threads"):
        return chirplet_reassign_parallel(x, n_jobs=n_jobs, backend=backend, **kw)
    if backend in ("gpu", "gpu-numpy"):
        kw.pop("workers", None)
        return chirplet_reassign_gpu(x, xp="cupy" if backend == "gpu" else "numpy",
                                     precision=precision, **kw)
    raise ValueError(f"unknown backend {backend!r}")


# =========================================================================
# Multi-core CPU
# =========================================================================
def _segment(x, a, b, hop, pad):
    """Samples needed for frames a..b-1 (frame i centred at i*hop), with `pad`
    samples (a multiple of hop) on each side; zeros before the start of x, exactly
    like the zero padding of chirplet_reassign."""
    s0, s1 = a * hop - pad, min((b - 1) * hop + pad + 1, x.size)
    seg = x[max(s0, 0):s1]
    if s0 < 0:
        seg = np.concatenate([np.zeros(-s0, dtype=x.dtype), seg])
    return seg


def _block(seg, kw, a, b, pad_frames, targets):
    """Compute one frame block with the serial code and store it into `targets`
    ({key: array} for threads, {key: (shm name, shape, dtype)} for processes)."""
    res = CR.chirplet_reassign(seg, **kw)
    sl = slice(pad_frames, pad_frames + (b - a))
    for key, tgt in targets.items():
        if isinstance(tgt, np.ndarray):
            tgt[a:b] = res[key][sl]
        else:
            name, shape, dtype = tgt
            shm = shared_memory.SharedMemory(name=name)
            try:
                np.ndarray(shape, dtype, buffer=shm.buf)[a:b] = res[key][sl]
            finally:
                shm.close()
    return b - a


def chirplet_reassign_parallel(x, n_jobs=None, backend="processes", blocks_per_job=4,
                               start_method=None, **kw):
    """Multi-core chirplet_reassign: frame blocks computed in parallel by the serial
    code. Same keywords / return dict as chirplet_reassign; equal results.
    n_jobs: workers (default: all cores). backend: "processes" or "threads".
    start_method: multiprocessing start method; default "fork" on Linux (fast
    start-up), "spawn" elsewhere."""
    x = np.asarray(x)
    n_jobs = n_jobs or os.cpu_count() or 1
    hop = kw.get("hop", 4)
    sigma = kw.get("sigma", 10.0)
    L = kw.get("window_length") or 2 * int(np.ceil(4 * sigma))
    pad_frames = math.ceil(L / hop) + 1
    pad = pad_frames * hop
    T = len(range(0, x.size, hop))
    kw = dict(kw, workers=1)                      # one FFT thread per worker

    meta = CR.chirplet_reassign(x[:1], **kw)      # grids, shapes and dtypes of the outputs
    keys = [k for k in meta if k in _OUT_KEYS]
    shapes = {k: (T,) + meta[k].shape[1:] for k in keys}

    n_blocks = max(1, min(T, n_jobs * blocks_per_job))
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    out = {k: v for k, v in meta.items() if k not in _OUT_KEYS}
    out["t"] = np.arange(0, x.size, hop)
    if backend == "threads" or n_jobs == 1:
        arrays = {k: np.empty(shapes[k], meta[k].dtype) for k in keys}
        with ThreadPoolExecutor(n_jobs) as ex:
            list(ex.map(lambda ab: _block(_segment(x, *ab, hop, pad), kw, *ab, pad_frames, arrays),
                        blocks))
        out.update(arrays)
        return out
    if backend != "processes":
        raise ValueError("backend must be 'processes' or 'threads'")

    shms = {k: shared_memory.SharedMemory(create=True, size=max(1, int(np.prod(shapes[k]))
                                                                 * meta[k].dtype.itemsize))
            for k in keys}
    try:
        targets = {k: (shms[k].name, shapes[k], meta[k].dtype) for k in keys}
        start_method = start_method or ("fork" if sys.platform.startswith("linux") else "spawn")
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context(start_method)) as ex:
            futs = [ex.submit(_block, _segment(x, a, b, hop, pad), kw, a, b, pad_frames, targets)
                    for a, b in blocks]
            for f in futs:
                f.result()                        # re-raise worker errors
        for k in keys:
            out[k] = np.ndarray(shapes[k], meta[k].dtype, buffer=shms[k].buf).copy()
    finally:
        for s in shms.values():
            s.close()
            s.unlink()
    return out


# =========================================================================
# GPU (CuPy) / vectorized
# =========================================================================
def chirplet_reassign_gpu(x, sigma=10.0, window_length=None, hop=4, nfft=128, n_beta_in=16,
                          beta_max=None, n_omega_out=256, n_beta_out=64, mode="joint",
                          gamma=1e-2, chunk=512, omega_range=None, return_raw=False,
                          return_C=False, return_complex=False, xp="cupy",
                          precision="single", compute_R=True):
    """chirplet_reassign on the GPU (xp="cupy") or as the same vectorized code on
    the CPU (xp="numpy"). precision: "single" (complex64; fastest on GPUs) or
    "double". Same keywords and return dict as chirplet_reassign (outputs are NumPy
    arrays, compute_R=False skips the reassignment); mode "two_step" is not supported."""
    if xp == "cupy":
        import cupy as xp
    elif xp == "numpy":
        xp = np
    if mode == "two_step":
        raise NotImplementedError("mode 'two_step' is only in the CPU implementations")
    cdt, rdt = (np.complex64, np.float32) if precision == "single" else (np.complex128, np.float64)
    x = np.asarray(x).astype(complex)
    L = int(window_length or 2 * int(np.ceil(4 * sigma)))
    if nfft < L:
        raise ValueError("nfft must be >= window_length")
    beta_max = np.pi / (4 * sigma) if beta_max is None else beta_max
    if mode == "fsst":
        n_beta_in = n_beta_out = 1
        beta_in = beta_out = np.zeros(1)
        d_b = 1.0
    else:
        beta_in = np.linspace(-beta_max, beta_max, n_beta_in)
        beta_out = np.linspace(-beta_max, beta_max, n_beta_out)
        d_b = beta_out[1] - beta_out[0]
    lo, hi = (-np.pi, np.pi) if omega_range is None else omega_range
    wrap = bool(np.isclose(hi - lo, TWO_PI))
    d_om = (hi - lo) / n_omega_out
    omega_out = lo + d_om * np.arange(n_omega_out)
    omega_in = TWO_PI * (np.arange(nfft) - nfft // 2) / nfft
    Ko, Bo = n_omega_out, n_beta_out
    cell = Ko * Bo

    tau = np.arange(L) - L // 2
    win = np.exp(-tau ** 2 / (2 * sigma ** 2)) / np.sqrt(sigma)
    base = win[None, :] * np.exp(-0.5j * beta_in[:, None] * tau[None, :] ** 2)
    need_ops = compute_R or return_raw
    mult = xp.asarray(np.stack([base, base * (-1j * tau), base * (-(tau ** 2))]) if need_ops
                      else base[None], dtype=cdt)
    pos = xp.asarray(tau % nfft)
    tau_d = xp.asarray(tau)
    W = xp.asarray(omega_in, dtype=rdt)[None, :, None]
    Bg = xp.asarray(beta_in, dtype=rdt)[None, None, :]
    s2 = sigma ** 2

    n0 = np.arange(0, x.size, hop)
    T = n0.size
    xpad = xp.asarray(np.pad(x, (L, L)), dtype=cdt)
    R = np.zeros((T, Ko, Bo) if compute_R else (0, Ko, Bo), np.float32)
    Rwo = np.zeros_like(R)
    Rwb = np.zeros_like(R)
    C2_all = np.zeros((T, nfft, n_beta_in), np.float32) if return_C else None
    C_all = np.zeros((T, nfft, n_beta_in), np.complex64) if return_complex else None
    raw_o = np.zeros((T, nfft, n_beta_in), np.float32) if return_raw else None
    raw_b = np.zeros((T, nfft, n_beta_in), np.float32) if return_raw else None
    to_host = (lambda a: a.get()) if xp is not np else (lambda a: a)
    errstate = (lambda: np.errstate(divide="ignore", invalid="ignore")) if xp is np else _Null

    for s in range(0, T, chunk):
        c = min(chunk, T - s)
        n0c = xp.asarray(n0[s:s + c])
        frames = xpad[n0c[:, None] + L + tau_d[None, :]]                        # (c, L)
        buf = xp.zeros((mult.shape[0], c, n_beta_in, nfft), dtype=cdt)
        buf[..., pos] = frames[None, :, None, :] * mult[:, None, :, :]
        F = xp.fft.fftshift(xp.fft.fft(buf, axis=-1), axes=-1)
        C = xp.swapaxes(F[0], 1, 2)                                             # (c, nfft, Nb)
        P = (C.real ** 2 + C.imag ** 2).astype(rdt)
        if return_C:
            C2_all[s:s + c] = to_host(P.astype(np.float32))
        if return_complex:
            C_all[s:s + c] = to_host(C.astype(np.complex64))
        if not need_ops:
            continue
        Cw, Cww = xp.swapaxes(F[1], 1, 2), xp.swapaxes(F[2], 1, 2)
        keep = (P >= gamma * P.max(axis=(1, 2), keepdims=True)) & (P > 0)
        with errstate():
            if mode == "fsst":
                b_hat = xp.zeros(C.shape, dtype=rdt)
                w_hat = W + (Cw / C).real / s2
            else:
                D = C * Cww - Cw * Cw
                b_hat = Bg + (C * C / D).imag
                if mode == "joint":
                    w_hat = W - (C * Cw / D).real
                elif mode == "eq6":
                    w_hat = W + (1 + s2 ** 2 * (Bg - b_hat) ** 2) / s2 * (Cw / C).real
                elif mode == "eq9":
                    step = beta_in[1] - beta_in[0] if n_beta_in > 1 else 1.0
                    bi = xp.rint((b_hat - beta_in[0]) / step)
                    bi = xp.clip(xp.where(xp.isfinite(bi), bi, 0), 0, n_beta_in - 1).astype(xp.int64)
                    w_hat = W + (xp.take_along_axis(Cw, bi, axis=2)
                                 / xp.take_along_axis(C, bi, axis=2)).real / s2
                else:
                    raise ValueError(mode)
            w_hat = (w_hat + np.pi) % TWO_PI - np.pi
            keep &= xp.isfinite(b_hat) & xp.isfinite(w_hat)

            # scatter-add into the output grid; invalid coefficients -> discarded bin
            ko = xp.rint((xp.where(keep, w_hat, 0) - lo) / d_om).astype(xp.int64)
            if wrap:
                ko %= Ko
            kb = (xp.rint((xp.where(keep, b_hat, 0) - beta_out[0]) / d_b).astype(xp.int64)
                  if Bo > 1 else xp.zeros(C.shape, dtype=xp.int64))
            ok = keep & (ko >= 0) & (ko < Ko) & (kb >= 0) & (kb < Bo)
            tl = xp.arange(c)[:, None, None]
            lin = xp.where(ok, (tl * Ko + ko) * Bo + kb, c * cell).ravel()
            e = xp.where(ok, P, 0).ravel()
            n_bins = c * cell + 1
            for target, wts in (((R, e), (Rwo, e * xp.where(ok, w_hat, 0).ravel()),
                                 (Rwb, e * xp.where(ok, b_hat, 0).ravel())) if compute_R else ()):
                acc = xp.bincount(lin, weights=wts, minlength=n_bins)[:-1]
                target[s:s + c] = to_host(acc.astype(np.float32)).reshape(c, Ko, Bo)
        if return_raw:
            raw_o[s:s + c] = to_host(w_hat.astype(np.float32))      # unmasked, as in the CPU code
            raw_b[s:s + c] = to_host(b_hat.astype(np.float32))

    out = dict(t=n0, omega_in=omega_in, beta_in=beta_in, omega_out=omega_out,
               beta_out=beta_out, sigma=sigma, hop=hop, d_om=d_om)
    if compute_R:
        out.update(R=R, Rw_omega=Rwo, Rw_beta=Rwb)
    if return_C:
        out["C2"] = C2_all
    if return_complex:
        out["C"] = C_all
    if return_raw:
        out["omega_hat"], out["beta_hat"] = raw_o, raw_b
    return out


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
