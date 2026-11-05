"""
Numba version of chirplet_ridge_fast.estimate_if:

    t, IF = estimate_if(x, **kwargs)

gives the same values, bit for bit, as

    t, IF, _ = chirplet_ridge.estimate_ifs(x, n_components=1, mode="eq9",
                                           onesided=False, ridge="dp",
                                           prediction=True, **kwargs)
    IF = IF[0]

kwargs are those of estimate_ifs, with its defaults. Used by
fmcw_aux.estimate_if_fcrm(..., numba=True).

Same algorithm as chirplet_ridge_fast (see its docstring), with the work in
compiled loops (numba) instead of numpy calls:
  * transform: per chunk of frames a kernel builds the FFT input, laid out
    (omega, beta) so that scipy's FFT (single-threaded, along axis -2) returns
    chirplet_reassign's layout directly; one kernel then does, frame by frame,
    the reassignment (the kept coefficients selected without branches, the
    operators in vectorizable passes), the cell sums R and Rw_omega, the gaussian
    blur of R, its block maxima and the near-tie candidates. Only the nonzero
    cells of each frame are kept (row, column, R, Rw_omega): the energy window
    and the read-out need nothing else.
  * both DP levels are single kernels: the forward pass computes max scores
    only, the traceback recomputes the first maximizing predecessor of the one
    state per frame on the path (as a dense argmax).
  * the energy window around the coarse path and the read-out cells are kernels.
The logs and the final float32 read-out sums stay numpy calls. All work arrays
are kept between calls (no page faults after the first call).

Exactness: the kernels repeat numpy's / scipy's arithmetic operation by
operation. Complex abs, multiply and square are numpy's SIMD formulas (they use
fused multiply-adds), complex division is numpy's (Smith's method with a
reciprocal), the blur is scipy.ndimage.correlate1d's symmetric-kernel loop
(x0 w0 + sum over j = r .. 1 of (x[-j] + x[+j]) w_j, along omega, then along beta),
sums accumulate in the same order, and the FFT along axis -2 is bit-identical to
the original's along the last axis. On first use self-checks compare these
pieces with numpy / scipy; if anything differs (another numpy / scipy version or
CPU), or numba is not installed, estimate_if falls back to
chirplet_ridge_fast.estimate_if (with a warning).

Options not covered (see chirplet_ridge_fast), non-finite x and |x| >= 1e100 fall
back the same way. The compiled kernels are cached on disk (__pycache__): the
first use after a change of this file compiles them (~20 s), later processes
only load them (~1 s).
"""
import math
import warnings

import numpy as np
import scipy.fft as sfft

from . import chirplet_ridge as CR
from . import chirplet_ridge_fast as CRF

try:
    from numba import njit
    from numba.core import cgutils, types
    from numba.extending import intrinsic
    from llvmlite import ir
    HAVE_NUMBA = True
except ImportError:                         # estimate_if falls back to chirplet_ridge_fast
    HAVE_NUMBA = False

TWO_PI = 2 * np.pi
_CHUNK = 8            # frames per FFT chunk
_PLAIN, _HOLD, _FLIP, _FREE = 0, 1, 2, 3     # kinds of DP moves (chirplet_ridge._viterbi_flips)


def estimate_if(x, **kwargs):
    """IF of the strongest ridge, the same values as
        t, IF, _ = chirplet_ridge.estimate_ifs(x, n_components=1, mode="eq9",
                                               onesided=False, ridge="dp",
                                               prediction=True, **kwargs)
        IF = IF[0]
    kwargs: those of estimate_ifs (sigma, hop, nfft, ..., zero_crossings,
    hold_sign, lam_b_cross, fs), with its defaults.
    Returns t (T,) in units of 1 / fs and IF (T,) in units of fs."""
    p = dict(CRF._DEFAULTS)
    p.update(kwargs)
    xa = np.asarray(x)
    # the kernels assume finite intermediate values: x finite and far from overflow
    if not (_numba_ok() and CRF._supported(xa, p, kwargs) and np.isfinite(xa).all()
            and float(np.abs(xa).max()) < 1e100
            and _fft_layout_ok(int(p["nfft"]), int(p["n_beta_in"]))):
        return CRF.estimate_if(x, **kwargs)
    fs = p["fs"]
    t, w = _estimate_if(xa.astype(np.float64, copy=False), p)
    return t / fs, w * fs / TWO_PI             # as chirplet_ridge.estimate_ifs


def release_buffers():
    """Free the work buffers and caches kept between calls."""
    _WORK.clear()
    _TABLES.clear()


# =========================================================================
# per-process state
# =========================================================================
_WORK = {}
_TABLES = {}
_STATE = {}


def _buf(name, shape, dtype=np.float64):
    """Work buffer `name` viewed as `shape` (uninitialized), reused between calls."""
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    size = int(np.prod(shape))
    b = _WORK.get(name)
    if b is None or b.dtype != np.dtype(dtype) or b.size < size:
        b = np.empty(max(1, size + size // 8), dtype)
        _WORK[name] = b
    return b[:size].reshape(shape)


def _gauss_weights(sigma, truncate=4.0):
    """Weights scipy.ndimage.gaussian_filter1d correlates with (order 0); empty if
    gaussian_filter skips the axis."""
    if not sigma > 1e-15:
        return np.zeros(0)
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    return np.ascontiguousarray((phi / phi.sum())[::-1])


# =========================================================================
# main
# =========================================================================
def _estimate_if(x, p):
    S = CRF._setup(p)
    hop = p["hop"]
    K, B = S.Ko, S.Bo
    r_k, r_b = p["r_k"], min(p["r_b"], B)
    wk, wb = _gauss_weights(p["blur"][0]), _gauss_weights(p["blur"][1])
    n0 = np.arange(0, x.size, hop)
    T = n0.size

    # ---- 1. transform, reassignment, coarse energy ----
    cells, foff, Ec, m_fr, Emax_fr = _transform(x, n0, S, p["gamma"], wk, wb, r_k, r_b)

    # ---- 2. coarse DP ----
    max_dk = p["max_dk"]
    if max_dk is None:
        max_dk = int(np.ceil(np.abs(S.beta_out).max() * hop / S.d_om)) + 6
    max_db = min(p["max_db"], B - 1)
    lam_k, lam_b, lam_b_cross = p["lam_k"], p["lam_b"], p["lam_b_cross"]
    flips = None
    if p["zero_crossings"] is not None:
        flips = CR._flip_frames(p["zero_crossings"], n0)
    kind, enter, leave = _moves(T, flips, p["hold_sign"])

    Kc, Bc = K // r_k, B // r_b
    if flips is None:
        side_c = np.ones(Kc * Bc, np.int8)                  # unused without flips
    else:   # side of a coarse cell (constant in time: no block straddles 0 Hz)
        side_c = np.repeat(S.side.reshape(Kc, r_k)[:, 0], Bc).astype(np.int8)
    offs, nof, Pw, pad = _coarse_tables(p, S, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross,
                                        max_dk, max_db, None if flips is None else side_c)
    pc = np.empty(T, np.int64)
    n_c = Kc * Bc
    best = _dp_coarse(Ec, kind, enter, leave, side_c, offs, nof, Pw, pad, pc,
                      _buf("dpc_Cp", (2, n_c + 2 * pad)), _buf("dpc_val", (2, n_c)),
                      _buf("dpc_Cu", (T, 2, n_c)))
    _check_dp(best, flips)
    ic, jc = np.divmod(pc, Bc)

    # ---- E on the blocks of the coarse path and their neighbours ----
    mk, mb = wk.size // 2, wb.size // 2
    nk, nb = 3 * r_k, 3 * r_b
    Ew = _buf("Ew", (T, nk, nb))
    _energy_window(cells[0], cells[1], cells[2], foff, (ic - 1) * r_k, (jc - 1) * r_b,
                   nk, nb, mk, mb, K, B, wk, wb, Ew)
    # _energy_log on these cells: log(blur(R) + a + 1e-300) - Emax (in place)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.add(Ew, (1e-6 * m_fr)[:, None, None], out=Ew)
        Ew += 1e-300
        np.log(Ew, out=Ew)
        np.subtract(Ew, Emax_fr[:, None, None], out=Ew)

    # ---- 3. fine DP (states k_cent + LK, b_cent + LB) ----
    n_f = (2 * r_k + 1) * (2 * r_b + 1)
    k_cent, b_cent = _buf("k_cent", T, np.int64), _buf("b_cent", T, np.int64)
    Ef, side_f = _buf("Ef", (T, n_f)), _buf("side_f", (T, n_f), np.int8)
    _fine_states(Ew, ic, jc, r_k, r_b, K, B, S.side.astype(np.int8), k_cent, b_cent, Ef, side_f)
    pf = np.empty(T, np.int64)
    best = _dp_fine(Ef, kind, enter, leave, side_f, k_cent, b_cent, S.shift, r_k, r_b, B,
                    float(lam_k), float(lam_b), float(lam_b_cross), int(max_dk), int(max_db), pf,
                    _buf("dpf_C", (2, n_f)), _buf("dpf_val", (2, n_f)),
                    _buf("dpf_Cu", (T, 2, n_f)),
                    _buf("dpf_Q", (2 * r_b + 1) * (4 * r_k + 1) * (2 * r_b + 1)))
    _check_dp(best, flips)
    kp = k_cent + pf // (2 * r_b + 1) - r_k
    bp = b_cent + pf % (2 * r_b + 1) - r_b

    # ---- 4. read-out ----
    w = _readout(cells, foff, S, kp, bp, *p["readout_cells"])
    return n0, w


def _check_dp(best, flips):
    if flips is not None and not np.isfinite(best):
        raise ValueError("no ridge satisfies the zero-crossing constraints (an interval "
                         "too short for max_dk, or hold_sign with a sign change elsewhere)")


def _moves(T, flips, hold_sign):
    """Kind of each move t-1 -> t, entering / leaving a zero-crossing interval
    (as chirplet_ridge._viterbi_flips)."""
    inside = np.zeros(T, bool)
    free = np.zeros(T, bool)
    for t1, t2 in (flips or []):
        inside[t1 + 1:t2] = True
        free[t1 + 1:t2 + 1] = True
    prev = np.r_[False, inside[:-1]]
    enter = inside & ~prev
    leave = prev & ~inside
    fixed = _HOLD if (hold_sign and flips is not None) else _PLAIN
    kind = np.where(~free, fixed, np.where(~(inside | prev), _FLIP, _FREE)).astype(np.int64)
    return kind, enter, leave


# =========================================================================
# 1. transform
# =========================================================================
def _transform(x, n0, S, gamma, wk, wb, r_k, r_b):
    """Returns
      cells : (omega row, beta column, R, Rw_omega) of the nonzero cells of every
              frame (R, Rw_omega: chirplet_reassign's float32 cell sums)
      foff  : index of each frame's first cell (T + 1)
      Ec    : ridge_dp's coarse energy (T, Kc Bc)
      m_fr, Emax_fr : per frame max of blur(R) and of its log energy"""
    L, nfft, Nb = S.L, S.nfft, S.Nb
    K, B = S.Ko, S.Bo
    Kc, Bc = K // r_k, B // r_b
    T = n0.size
    cap = T * min(nfft * Nb, K * B)            # a frame has at most this many cells
    crow, ccol = _buf("crow", cap, np.int32), _buf("ccol", cap, np.int32)
    cR, cRw = _buf("cR", cap, np.float32), _buf("cRw", cap, np.float32)
    foff = np.empty(T + 1, np.int64)
    rbm = _buf("rbm", (T, Kc, Bc))
    m_fr = np.empty(T)
    small = np.zeros(T, np.int8)
    ncap = max(4096, _WORK.get("near_v", np.zeros(0)).size)
    near_i, near_v = _buf("near_i", (ncap, 3), np.int64), _buf("near_v", ncap)
    bufc = _buf("fftbuf", 3 * _CHUNK * Nb * nfft, complex)
    P = _buf("P", (nfft, Nb))
    R64, R64w = _buf("R64", K * B), _buf("R64w", K * B)   # zero between frames
    R64[:] = 0.0
    R64w[:] = 0.0
    rownz, act = _buf("rownz", K, np.int64), _buf("act", K, np.int8)
    rownz[:] = 0
    act[:] = 0
    touched, trow = _buf("touched", K * B, np.int64), _buf("trow", K * B, np.int64)
    Ww = _buf("W", (10, Nb * nfft + 24))       # rows 16 kB + 192 B apart
    KI = _buf("KI", (2, Nb * nfft), np.int64)
    KO, KB = _buf("KO", Nb * nfft, np.int64), _buf("KB", Nb * nfft, np.int64)
    OK = _buf("OK", Nb * nfft, np.bool_)
    o1, o2 = _buf("o1", (K, B)), _buf("o2", (K, B))
    Rsmall = _buf("Rsmall", (_CHUNK, K, B))
    lm_small = {}
    full_circle = bool(S.full_circle)
    multT = np.ascontiguousarray(S.mult.transpose(0, 2, 1))      # (3, L, Nb)
    n_tot = n_near = 0
    for s in range(0, T, _CHUNK):
        c = min(_CHUNK, T - s)
        buf = bufc[:3 * c * Nb * nfft].reshape(3, c, nfft, Nb)
        _prep(x, n0, s, c, L, S.n_neg, multT, buf)
        F = sfft.fft(buf, axis=-2, workers=1, overwrite_x=True)
        while True:
            n_tot2, n_near2 = _post(F, s, c, nfft // 2, gamma, S.beta_in, S.omega_in, S.s2,
                                    S.b_step, S.lo, S.d_om, full_circle, K, B,
                                    S.beta_out[0], S.d_b, wk, wb, r_k, r_b,
                                    crow, ccol, cR, cRw, n_tot, foff, rbm, m_fr, small,
                                    near_i, near_v, n_near, P, R64, R64w, touched, trow,
                                    rownz, act, Ww, KI, KO, KB, OK, o1, o2, Rsmall)
            if n_near2 <= near_v.size:
                break
            ncap = 2 * n_near2                 # (rare) more near-tie cells than room: redo
            old_i, old_v = near_i[:n_near].copy(), near_v[:n_near].copy()
            near_i, near_v = _buf("near_i", (ncap, 3), np.int64), _buf("near_v", ncap)
            near_i[:n_near], near_v[:n_near] = old_i, old_v
        n_tot, n_near = n_tot2, n_near2
        for f in np.flatnonzero(small[s:s + c]):
            # max(blur(R)) <= ~1e-274: the log of every cell, as the original
            t = s + f
            X = Rsmall[f] + 1e-6 * m_fr[t]
            X += 1e-300
            with np.errstate(divide="ignore"):
                np.log(X, out=X)
            lm_small[t] = X.reshape(Kc, r_k, B).max(axis=1).reshape(Kc, Bc, r_b).max(axis=-1)
    foff[T] = n_tot

    # _energy_log + ridge_dp's block max: log(block max of blur(R) + a) (log and
    # x -> x + a are monotone), with the near-tie cells' own logs as a guard
    a = 1e-6 * m_fr
    lm = _buf("lm", (T, Kc, Bc))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.add(rbm, a[:, None, None], out=lm)
        np.log(lm, out=lm)
        if n_near:
            ni = near_i[:n_near]
            np.maximum.at(lm, (ni[:, 0], ni[:, 1], ni[:, 2]),
                          np.log(near_v[:n_near] + a[ni[:, 0]]))
    for t, v in lm_small.items():
        lm[t] = v
    Em = lm.max(axis=(1, 2))
    Ec = _buf("Ec", (T, Kc * Bc))
    np.subtract(lm.reshape(T, -1), Em[:, None], out=Ec)
    cells = (crow[:n_tot], ccol[:n_tot], cR[:n_tot], cRw[:n_tot])
    return cells, foff, Ec, m_fr, Em


def _readout(cells, foff, S, kp, bp, dk=2, db=2):
    """chirplet_ridge.readout, IF part (rad/sample)."""
    K, B = S.Ko, S.Bo
    T = kp.size
    nr, ncol = 2 * dk + 1, 2 * db + 1
    R5, Ro5 = _buf("R5", (T, nr * ncol), np.float32), _buf("Ro5", (T, nr * ncol), np.float32)
    _readout_sums(cells[0], cells[1], cells[2], cells[3], foff, kp, bp, dk, db, K, R5, Ro5)
    full = (bp - db >= 0) & (bp + db + 1 <= B)
    e = R5.sum(axis=1)                         # (rows are summed independently)
    o = Ro5.sum(axis=1)
    for t in np.nonzero(~full)[0]:             # ridge at the edge of the beta grid
        j = np.arange(max(0, bp[t] - db), min(B, bp[t] + db + 1)) - (bp[t] - db)
        ix = np.ix_(np.arange(nr), j)
        e[t] = R5[t].reshape(nr, ncol)[ix].sum()
        o[t] = Ro5[t].reshape(nr, ncol)[ix].sum()
    w = np.empty(T)
    pos = e > 0
    w[pos] = o[pos] / e[pos]
    w[~pos] = S.omega_out[kp[~pos]]
    return w


def _coarse_tables(p, S, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross, max_dk, max_db, side):
    """Per kind of move: the offsets o = j - s of the allowed predecessors j of the
    coarse states s and Pw[o, s] = penalty of j -> s (inf: not allowed)."""
    key = (tuple(p[k] for k in ("sigma", "hop", "nfft", "n_beta_in", "beta_max_hz_s",
                                "n_omega_out", "n_beta_out", "fs")),
           Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross, max_dk, max_db,
           None if side is None else side.tobytes())
    tabs = _TABLES.get(key)
    if tabs is not None:
        return tabs
    Pc, Pc_free = CRF._coarse_pen(S.shift, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross,
                                  max_dk, max_db)
    if side is None:
        mats = [Pc]                                       # _PLAIN
    else:
        same = side[:, None] == side[None, :]
        mats = [Pc, np.where(same, Pc, np.inf),           # _PLAIN, _HOLD
                np.where(same, np.inf, Pc_free), Pc_free]  # _FLIP, _FREE
    n = Kc * Bc
    s = np.arange(n)
    offl = []
    for P in mats:
        r, q = np.nonzero(np.isfinite(P))
        offl.append(np.unique(q - r))
    nof = np.array([o.size for o in offl], np.int64)
    offs = np.zeros((len(mats), max(1, nof.max())), np.int64)
    Pw = np.full((len(mats), offs.shape[1], n), np.inf)
    for i, (P, o) in enumerate(zip(mats, offl)):
        offs[i, :o.size] = o
        j = s[None, :] + o[:, None]
        inr = (j >= 0) & (j < n)
        Pw[i, :o.size] = np.where(inr, P[s[None, :], np.where(inr, j, 0)], np.inf)
    pad = int(np.abs(offs).max())
    tabs = _TABLES[key] = (offs, nof, Pw, pad)
    return tabs


# =========================================================================
# self-check: the kernels' arithmetic against numpy / scipy on this machine
# =========================================================================
def _numba_ok():
    ok = _STATE.get("ok")
    if ok is None:
        ok = _STATE["ok"] = _self_check()
    return ok


def _fft_layout_ok(nfft, Nb):
    """scipy's FFT along axis -2 of (3, c, nfft, Nb) bit-identical to the
    original's along the last axis of (3, c, Nb, nfft)? (checked once per size)"""
    key = ("fft", nfft, Nb)
    ok = _STATE.get(key)
    if ok is None:
        rng = np.random.default_rng(nfft * 1000 + Nb)
        a = rng.standard_normal((3, 3, Nb, nfft)) + 1j * rng.standard_normal((3, 3, Nb, nfft))
        a[..., nfft // 4:nfft // 2] = 0.0
        f_last = sfft.fft(a, axis=-1, workers=1)
        f_mid = sfft.fft(np.ascontiguousarray(a.transpose(0, 1, 3, 2)), axis=-2, workers=1)
        ok = _STATE[key] = bool(np.array_equal(
            f_last.view(np.uint64), np.ascontiguousarray(f_mid.transpose(0, 1, 3, 2)).view(np.uint64)))
        if not ok:
            warnings.warn("scipy's FFT along axis -2 differs from axis -1 on this machine: "
                          "chirplet_ridge_numba falls back to chirplet_ridge_fast",
                          RuntimeWarning, stacklevel=3)
    return ok


def _self_check():
    if not HAVE_NUMBA:
        warnings.warn("numba is not installed: chirplet_ridge_numba falls back to "
                      "chirplet_ridge_fast", RuntimeWarning, stacklevel=3)
        return False
    try:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(20240821)
        n = 20000
        mag = 10.0 ** rng.uniform(-20, 20, (4, n))
        z1 = rng.standard_normal(n) * mag[0] + 1j * rng.standard_normal(n) * mag[1]
        z2 = rng.standard_normal(n) * mag[2] + 1j * rng.standard_normal(n) * mag[3]
        z1[:6] = [0, 3, 2j, -0.0 + 1j, 1e-310 + 1e-310j, 1 + 1e-300j]
        z2[:6] = [1, 0.5j, 3, 2 - 2j, 1e-300 + 0j, 1e-308 - 1e300j]
        w = rng.uniform(-50, 50, n)
        got = _arith_kernels(z1, z2, w)
        ref = (np.abs(z1), np.abs(z1), z1 * z2, z1 ** 2, z1 / z2, z1 / z2,
               (w + np.pi) % TWO_PI - np.pi)
        ok = all(np.array_equal(g.view(np.uint64), r.view(np.uint64)) for g, r in zip(got, ref))
        for sig, shape in (((1.0, 1.0), (60, 40)), ((1.0, 1.0), (30, 6)),
                           ((0.8, 1.6), (60, 40)), ((0.0, 1.0), (20, 12))):
            R = (rng.random(shape) * (rng.random(shape) < 0.2)
                 * 10.0 ** rng.uniform(-3, 3, shape)).astype(np.float32)
            o1, o2 = np.empty(shape), np.empty(shape)
            _blur(R.astype(np.float64), np.ones(shape[0], np.int8),
                  _gauss_weights(sig[0]), _gauss_weights(sig[1]), o1, o2)
            refb = gaussian_filter(R[None], (0,) + sig, mode="nearest", output=np.float64)[0]
            ok &= np.array_equal(o2.view(np.uint64), refb.view(np.uint64))
    except Exception as ex:                    # pragma: no cover
        warnings.warn(f"chirplet_ridge_numba self-check failed ({type(ex).__name__}: {ex}): "
                      "falling back to chirplet_ridge_fast", RuntimeWarning, stacklevel=3)
        return False
    if not ok:
        warnings.warn("numba arithmetic differs from numpy/scipy on this machine: "
                      "chirplet_ridge_numba falls back to chirplet_ridge_fast",
                      RuntimeWarning, stacklevel=3)
    return bool(ok)


# =========================================================================
# kernels
# =========================================================================
if HAVE_NUMBA:
    _jit = njit(cache=True, error_model="numpy")

    @intrinsic
    def _fma(typingctx, a, b, c):
        """Fused multiply-add a * b + c (one rounding), llvm.fma.f64."""
        if not all(isinstance(v, types.Float) for v in (a, b, c)):
            return None
        sig = types.float64(types.float64, types.float64, types.float64)

        def codegen(context, builder, signature, args):
            fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * 3)
            fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.fma.f64")
            return builder.call(fn, args)
        return sig, codegen

    @_jit
    def _cabs(re, im):
        """np.abs(complex128): numpy's SIMD loop, larger * sqrt(1 + ratio^2)."""
        re, im = abs(re), abs(im)
        if re == np.inf or im == np.inf:
            return np.inf
        if re != re or im != im:
            return np.nan
        lg, sm = max(re, im), min(im, re)
        r = sm / lg if lg != 0.0 else 0.0
        return math.sqrt(_fma(r, r, 1.0)) * lg

    @_jit
    def _cabs_finite(re, im):
        """_cabs for finite re, im, without branches (vectorizes)."""
        re, im = abs(re), abs(im)
        lg = re if re > im else im
        sm = im if im < re else re
        r = sm / lg if lg != 0.0 else 0.0
        return math.sqrt(_fma(r, r, 1.0)) * lg

    @_jit
    def _cmul(ar, ai, br, bi):
        """complex128 multiply, numpy's SIMD loop (a_re * b -/+ a_im * b_swapped, fused)."""
        return _fma(ar, br, -(ai * bi)), _fma(ar, bi, ai * br)

    @_jit
    def _cdiv(ar, ai, br, bi):
        """complex128 divide, numpy's loop (Smith's method with a reciprocal)."""
        abr, abi = abs(br), abs(bi)
        if abr >= abi:
            if abr == 0 and abi == 0:
                return ar / abr, ai / abr
            rat = bi / br
            scl = 1.0 / (br + bi * rat)
            return (ar + ai * rat) * scl, (ai - ar * rat) * scl
        rat = br / bi
        scl = 1.0 / (bi + br * rat)
        return (ar * rat + ai) * scl, (ai * rat - ar) * scl

    @_jit
    def _cdiv_nz(ar, ai, br, bi):
        """_cdiv without branches (selects; vectorizes); wrong for br == bi == 0,
        which the callers check for."""
        big = abs(br) >= abs(bi)
        p = br if big else bi
        q = bi if big else br
        rat = q / p
        scl = 1.0 / (p + q * rat)
        re = (ar + ai * rat) if big else (ar * rat + ai)
        im = (ai - ar * rat) if big else (ai * rat - ar)
        return re * scl, im * scl

    @_jit
    def _wrap(w):
        """(w + pi) % 2pi - pi as numpy (np.remainder: the remainder of x in [0, 2pi)
        is x itself, exactly; otherwise fmod-based, as numpy)."""
        x = w + np.pi
        if x >= 0.0 and x < TWO_PI:
            return x - np.pi
        return x % TWO_PI - np.pi

    @_jit
    def _arith_kernels(z1, z2, w):
        n = z1.size
        ab, ab2, mu, sq, dv, dv2, wr = (np.empty(n), np.empty(n), np.empty(n, np.complex128),
                                        np.empty(n, np.complex128), np.empty(n, np.complex128),
                                        np.empty(n, np.complex128), np.empty(n))
        for i in range(n):
            a, b = z1[i], z2[i]
            ab[i] = _cabs(a.real, a.imag)
            ab2[i] = _cabs_finite(a.real, a.imag)
            r, m = _cmul(a.real, a.imag, b.real, b.imag)
            mu[i] = complex(r, m)
            r, m = _cmul(a.real, a.imag, a.real, a.imag)
            sq[i] = complex(r, m)
            r, m = _cdiv(a.real, a.imag, b.real, b.imag)
            dv[i] = complex(r, m)
            r, m = _cdiv_nz(a.real, a.imag, b.real, b.imag)
            dv2[i] = complex(r, m)
            wr[i] = _wrap(w[i])
        return ab, ab2, mu, sq, dv, dv2, wr

    # Indexing note: numba wraps negative indices (Python semantics), which turns
    # an index LLVM cannot prove non-negative (e.g. b - 4) into a gather and
    # stops vectorization; the inner loops below index with i + const, i >= 0.

    @_jit
    def _blur(src, act, wk, wb, o1, o2):
        """scipy.ndimage.gaussian_filter(src, (sk, sb), mode="nearest", output=float64)
        of a 2-D float64 array holding float32 values (weights wk, wb from
        _gauss_weights; empty: axis skipped). ndimage.correlate1d with a symmetric
        kernel: out = x0 w0, then out += (x[-j] + x[+j]) w_j for j = r .. 1, in
        double; axis 0, then axis 1. Rows with act[k] == 0 (src zero within the
        radius of row k) are 0 in the output, as in the full computation."""
        if wk.size == 9 and wb.size == 9:
            _blur_r4(src, act, wk, wb, o1, o2)
        else:
            _blur_gen(src, act, wk, wb, o1, o2)

    @_jit
    def _blur_r4(src, act, wk, wb, o1, o2):
        """_blur for radius 4 on both axes (blur = (1, 1)): the stencil in registers."""
        n, m = src.shape
        a0, a1, a2, a3, a4 = wk[4], wk[3], wk[2], wk[1], wk[0]
        c0, c1, c2, c3, c4 = wb[4], wb[3], wb[2], wb[1], wb[0]
        for k in range(n):
            if act[k] == 0:
                continue
            m4, p4 = max(k - 4, 0), min(k + 4, n - 1)
            m3, p3 = max(k - 3, 0), min(k + 3, n - 1)
            m2, p2 = max(k - 2, 0), min(k + 2, n - 1)
            m1, p1 = max(k - 1, 0), min(k + 1, n - 1)
            for b in range(m):
                acc = src[k, b] * a0
                acc += (src[m4, b] + src[p4, b]) * a4
                acc += (src[m3, b] + src[p3, b]) * a3
                acc += (src[m2, b] + src[p2, b]) * a2
                acc += (src[m1, b] + src[p1, b]) * a1
                o1[k, b] = acc
        for k in range(n):
            if act[k] == 0:
                for b in range(m):
                    o2[k, b] = 0.0
                continue
            for i in range(m - 8):                                    # interior
                acc = o1[k, i + 4] * c0
                acc += (o1[k, i] + o1[k, i + 8]) * c4
                acc += (o1[k, i + 1] + o1[k, i + 7]) * c3
                acc += (o1[k, i + 2] + o1[k, i + 6]) * c2
                acc += (o1[k, i + 3] + o1[k, i + 5]) * c1
                o2[k, i + 4] = acc
            for e in range(m if m < 8 else 8):                        # edges
                b = e if (m < 8 or e < 4) else m - 8 + e
                acc = o1[k, b] * c0
                acc += (o1[k, max(b - 4, 0)] + o1[k, min(b + 4, m - 1)]) * c4
                acc += (o1[k, max(b - 3, 0)] + o1[k, min(b + 3, m - 1)]) * c3
                acc += (o1[k, max(b - 2, 0)] + o1[k, min(b + 2, m - 1)]) * c2
                acc += (o1[k, max(b - 1, 0)] + o1[k, min(b + 1, m - 1)]) * c1
                o2[k, b] = acc

    @_jit
    def _blur_gen(src, act, wk, wb, o1, o2):
        """_blur for any radius (one pass per tap, accumulated in the output rows)."""
        n, m = src.shape
        rk, rb = wk.size // 2, wb.size // 2
        for k in range(n):
            if act[k] == 0:
                continue
            y = o1[k]
            x0 = src[k]
            if wk.size == 0:
                for b in range(m):
                    y[b] = x0[b]
                continue
            w0 = wk[rk]
            for b in range(m):
                y[b] = x0[b] * w0
            for j in range(rk, 0, -1):
                xm, xp = src[max(k - j, 0)], src[min(k + j, n - 1)]
                wj = wk[rk - j]
                for b in range(m):
                    y[b] += (xm[b] + xp[b]) * wj
        for k in range(n):
            z = o2[k]
            if act[k] == 0:
                for b in range(m):
                    z[b] = 0.0
                continue
            y = o1[k]
            if wb.size == 0:
                for b in range(m):
                    z[b] = y[b]
                continue
            w0 = wb[rb]
            for b in range(m):
                z[b] = y[b] * w0
            for j in range(rb, 0, -1):
                wj = wb[rb - j]
                if m > 2 * j:                                         # interior
                    zi, yl, yr = z[j:m - j], y[0:m - 2 * j], y[2 * j:m]
                    for i in range(m - 2 * j):
                        zi[i] += (yl[i] + yr[i]) * wj
                    e1, e2 = j, m - j
                else:
                    e1, e2 = m, m
                for b in range(e1):                                   # edges
                    z[b] += (y[max(b - j, 0)] + y[min(b + j, m - 1)]) * wj
                for b in range(e2, m):
                    z[b] += (y[max(b - j, 0)] + y[min(b + j, m - 1)]) * wj

    @_jit
    def _prep(x, n0, s0, c, L, nn, multT, buf):
        """FFT input of frames s0 .. s0 + c - 1, laid out (3, c, nfft, Nb) for an
        FFT along axis -2 (bit-identical to the original's along the last axis of
        (3, c, Nb, nfft), and its output is already (omega, beta)):
        x[n0 + tau] * mult at row tau % nfft (multT = mult as (3, L, Nb); x real:
        complex multiply with imaginary part 0, as numpy)."""
        N = x.size
        nq, Nb = multT.shape[0], multT.shape[2]
        nfft = buf.shape[2]
        half = L // 2
        for f in range(c):
            t0 = n0[s0 + f] - half
            for q in range(nq):
                for i in range(L - nn, nfft - nn):
                    for b in range(Nb):
                        buf[q, f, i, b] = 0.0
                for i in range(L):
                    j = t0 + i
                    xr = x[j] if (j >= 0 and j < N) else 0.0
                    pos = i - nn if i >= nn else nfft - nn + i
                    for b in range(Nb):
                        m = multT[q, i, b]
                        r, im = _cmul(xr, 0.0, m.real, m.imag)
                        buf[q, f, pos, b] = complex(r, im)

    @_jit
    def _post(F, s0, c, h, gamma, beta_in, omega_in, s2, b_step, lo, d_om, full_circle,
              K, B, bmin, d_b, wk, wb, r_k, r_b,
              crow, ccol, cR, cRw, n_tot, foff, rbm, m_fr, small, near_i, near_v, n_near,
              P, R64, R64w, touched, trow, rownz, act, W, KI, KO, KB, OK, o1, o2, Rsmall):
        """Per frame of the chunk: reassignment (chirplet_reassign, mode eq9) of the
        kept coefficients, R and Rw_omega (float32 cell sums of e and e w), blur(R),
        its block maxima and the near-tie cells. The frame's nonzero cells go to the
        lists crow, ccol (omega row, beta column), cR, cRw from n_tot on (foff[t]:
        start of frame t). Returns the new cell and near-tie counts.
        F: FFT output (3, c, nfft, Nb), rows in FFT order (fftshift: row kmap[k]).
        Work arrays (passed in: allocating them per call costs page faults):
        P (nfft, Nb); R64, R64w (K * B), rownz (K), act (K): zero on entry and on
        return;
        touched, trow (K * B); W (10, > Nb nfft) with rows not 4 kB apart;
        KI (2, Nb nfft); KO, KB, OK (Nb nfft); o1, o2 (K, B); Rsmall (chunk, K, B)."""
        nfft, Nb = F.shape[2], F.shape[3]
        Kc, Bc = K // r_k, B // r_b
        ncap = near_v.size
        b0 = beta_in[0]
        rk = wk.size // 2
        R2 = R64.reshape(K, B)
        cm = np.empty(B)
        colv, lov = np.empty(B), np.empty(B)
        lanes = np.empty(8)
        kmap = np.empty(nfft, np.int64)            # fftshifted bin k -> FFT row
        for k in range(nfft):
            kmap[k] = k - h if k >= h else k + nfft - h
        for f in range(c):
            t = s0 + f
            foff[t] = n_tot
            G = F[:, f]                            # C, C_w, C_ww: (3, nfft, Nb)
            for k in range(nfft):
                g0 = G[0, kmap[k]]
                for b in range(Nb):
                    z = g0[b]
                    a = _cabs_finite(z.real, z.imag)
                    P[k, b] = a * a
            Pf = P.reshape(Nb * nfft)
            for l in range(8):
                lanes[l] = -1.0
            for i0 in range(0, Pf.size - 7, 8):    # max in 8 lanes (vectorizes)
                for l in range(8):
                    v = Pf[i0 + l]
                    lanes[l] = v if v > lanes[l] else lanes[l]
            pmax = -1.0
            for l in range(8):
                pmax = lanes[l] if lanes[l] > pmax else pmax
            for i in range(Pf.size - Pf.size % 8, Pf.size):
                pmax = Pf[i] if Pf[i] > pmax else pmax
            thr = gamma * pmax
            # The kept coefficients in chirplet_reassign's order (omega bin, then
            # beta), selected without branches (the keep test is unpredictable);
            # then the reassignment operators in passes over them (the divisions
            # vectorize), then the bins and the accumulation.
            m = 0
            for k in range(nfft):                  # fftshifted order
                for b in range(Nb):
                    p = P[k, b]
                    KI[0, m] = k
                    KI[1, m] = b
                    m += (p >= thr) & (p > 0.0)
            cr, ci, cwr, cwi = W[0, :m], W[1, :m], W[2, :m], W[3, :m]
            cwwr, cwwi, bb, pp = W[4, :m], W[5, :m], W[6, :m], W[7, :m]
            bh_a, wh_a = W[8, :m], W[9, :m]
            for i in range(m):
                k, b = KI[0, i], KI[1, i]
                kr = kmap[k]
                z, zw, zww = G[0, kr, b], G[1, kr, b], G[2, kr, b]
                cr[i], ci[i], cwr[i], cwi[i] = z.real, z.imag, zw.real, zw.imag
                cwwr[i], cwwi[i] = zww.real, zww.imag
                bb[i], pp[i] = beta_in[b], P[k, b]
            # b_hat = beta + Im(C ** 2 / D), D = C Cww - Cw ** 2
            zero = False
            for i in range(m):
                m1r, m1i = _cmul(cr[i], ci[i], cwwr[i], cwwi[i])
                s1r, s1i = _cmul(cwr[i], cwi[i], cwr[i], cwi[i])
                c2r, c2i = _cmul(cr[i], ci[i], cr[i], ci[i])
                dre, dim = m1r - s1r, m1i - s1i
                qr, qi = _cdiv_nz(c2r, c2i, dre, dim)
                zero |= (dre == 0.0) & (dim == 0.0)
                bh_a[i] = bb[i] + qi
            if zero:                               # D == 0: numpy's special case
                for i in range(m):
                    m1r, m1i = _cmul(cr[i], ci[i], cwwr[i], cwwi[i])
                    s1r, s1i = _cmul(cwr[i], cwi[i], cwr[i], cwi[i])
                    if m1r - s1r == 0.0 and m1i - s1i == 0.0:
                        c2r, c2i = _cmul(cr[i], ci[i], cr[i], ci[i])
                        qr, qi = _cdiv(c2r, c2i, m1r - s1r, m1i - s1i)
                        bh_a[i] = bb[i] + qi
            # C, Cw at the beta_in bin nearest to b_hat (eq9)
            for i in range(m):
                rr = np.rint((bh_a[i] - b0) / b_step)
                bi = 0 if rr != rr else int(min(max(rr, 0.0), Nb - 1.0))
                kr = kmap[KI[0, i]]
                zg, zwg = G[0, kr, bi], G[1, kr, bi]
                cr[i], ci[i], cwr[i], cwi[i] = zg.real, zg.imag, zwg.real, zwg.imag
                cwwr[i] = omega_in[KI[0, i]]
            # omega_hat = omega + Re(Cw / C) / sigma ** 2
            zero = False
            for i in range(m):
                dre, dim = _cdiv_nz(cwr[i], cwi[i], cr[i], ci[i])
                zero |= (cr[i] == 0.0) & (ci[i] == 0.0)
                wh_a[i] = cwwr[i] + dre / s2
            if zero:
                for i in range(m):
                    if cr[i] == 0.0 and ci[i] == 0.0:
                        dre, dim = _cdiv(cwr[i], cwi[i], cr[i], ci[i])
                        wh_a[i] = cwwr[i] + dre / s2
            # wrap omega_hat to [-pi, pi): (w + pi) % 2pi - pi; the remainder of
            # x in [0, 2pi) is x itself (exactly), others (rare) via _wrap
            wr_a = W[5, :m]                        # (row free by now)
            slow = False
            for i in range(m):
                x = wh_a[i] + np.pi
                slow |= not ((x >= 0.0) & (x < TWO_PI))
                wr_a[i] = x - np.pi
            if slow:
                for i in range(m):
                    x = wh_a[i] + np.pi
                    if not (x >= 0.0 and x < TWO_PI):
                        wr_a[i] = _wrap(wh_a[i])
            # finite checks and bins (_accumulate), without branches
            for i in range(m):
                bh, wh = bh_a[i], wr_a[i]
                ok = np.isfinite(bh) & np.isfinite(wh)
                kof = np.rint(((wh if ok else lo) - lo) / d_om)
                kbf = np.rint((bh - bmin) / d_b)
                okb = (kbf >= 0.0) & (kbf < B)
                KO[i] = int(kof)
                KB[i] = int(kbf if (ok & okb) else 0.0)
                OK[i] = ok & okb
            for i in range(m):
                ko = KO[i]
                if ko < 0 or ko >= K:              # (rare; avoids an integer division)
                    if full_circle:
                        KO[i] = ko % K
                    else:
                        KO[i] = 0
                        OK[i] = False
                        KB[i] = 0
            # cell sums of e and e w in float64, in the coefficient order (as the
            # original's bincount). A coefficient that is not kept adds +0.0 to a
            # valid cell, which changes nothing.
            nt = 0
            for i in range(m):
                ok = OK[i]
                ko, kb = KO[i], KB[i]
                cell = ko * B + kb
                touched[nt] = cell
                trow[nt] = ko
                nt += ok & (R64[cell] == 0.0)
                R64[cell] += pp[i] if ok else 0.0
                R64w[cell] += pp[i] * wr_a[i] if ok else 0.0
            # R, Rw_omega = float32 cell sums (R kept as float64 values for the blur);
            # the frame's cells for the energy window and the read-out
            for i in range(nt):
                cell = touched[i]
                r32 = np.float32(R64[cell])
                R64[cell] = np.float64(r32)
                rownz[trow[i]] = 1
                crow[n_tot] = trow[i]
                ccol[n_tot] = cell - trow[i] * B
                cR[n_tot] = r32
                cRw[n_tot] = np.float32(R64w[cell])
                R64w[cell] = 0.0
                n_tot += 1
            for k in range(K):                     # rows of blur(R) that can be nonzero
                for d in range(max(k - rk, 0), min(k + rk, K - 1) + 1):
                    if rownz[d]:
                        act[k] = 1
                        break
            _blur(R2, act, wk, wb, o1, o2)
            for i in range(nt):
                R64[touched[i]] = 0.0
                rownz[trow[i]] = 0
            for k in range(K):
                act[k] = 0
            # block maxima (column maxima over the r_k rows, then over r_b columns)
            m = -np.inf
            for kc in range(Kc):
                r0 = kc * r_k
                for b in range(B):
                    cm[b] = o2[r0, b]
                for i in range(1, r_k):
                    row = o2[r0 + i]
                    for b in range(B):
                        v = row[b]
                        cm[b] = v if v > cm[b] else cm[b]
                for bc in range(Bc):
                    blk = cm[bc * r_b:bc * r_b + r_b]
                    mx = blk[0]
                    for j in range(1, r_b):
                        if blk[j] > mx:
                            mx = blk[j]
                    rbm[t, kc, bc] = mx
                    if mx > m:
                        m = mx
            m_fr[t] = m
            a = 1e-6 * m
            if not a > 1e-280:
                small[t] = 1
                Rsmall[f, :, :] = o2
                continue
            # near-tie cells: col - 2e-9 (col + a) <= v < col (col: block max)
            for kc in range(Kc):
                for bc in range(Bc):
                    col = rbm[t, kc, bc]
                    lo2 = col - 2e-9 * (col + a)
                    for j in range(bc * r_b, bc * r_b + r_b):
                        colv[j] = col
                        lov[j] = lo2
                cnt = 0
                for i in range(r_k):
                    row = o2[kc * r_k + i]
                    for b in range(B):
                        v = row[b]
                        cnt += (v >= lov[b]) & (v < colv[b])   # no branches: vectorizes
                if cnt == 0:
                    continue
                for i in range(r_k):
                    row = o2[kc * r_k + i]
                    for b in range(B):
                        v = row[b]
                        if v >= lov[b] and v < colv[b]:
                            if n_near < ncap:
                                near_i[n_near, 0] = t
                                near_i[n_near, 1] = kc
                                near_i[n_near, 2] = b // r_b
                                near_v[n_near] = v
                            n_near += 1
        return n_tot, n_near

    @_jit
    def _dp_coarse(E, kind, enter, leave, side, offs, nof, Pw, pad, path, Cp, val, Cu):
        """chirplet_ridge._viterbi_flips / _viterbi on the coarse states (allowed
        predecessors of state s: j = s + o, o in offs[kind]). The forward pass only
        computes the max scores (4 predecessors per sweep over the states; a max
        does not depend on the order); the traceback recomputes, for the one state
        per frame on the path, the first maximizing predecessor (increasing j, as a
        dense argmax) from Cu[t, copy] = the scores the move t-1 -> t started from.
        Writes the path, returns the best final score.
        Work arrays: Cp (2, n + 2 pad), val (2, n), Cu (T, 2, n)."""
        T, n = E.shape
        NEG = -np.inf
        Cp[:, :] = NEG
        for s in range(n):
            Cp[0, pad + s] = E[0, s]
        nc = 1
        for t in range(1, T):
            kd = kind[t]
            if enter[t]:                           # split by side at t - 1
                for s in range(n):
                    v = Cp[0, pad + s]
                    Cp[0, pad + s] = v if side[s] == 1 else NEG
                    Cp[1, pad + s] = v if side[s] == -1 else NEG
                nc = 2
            no = nof[kd]
            for c in range(nc):
                cu = Cu[t, c]
                for s in range(n):
                    cu[s] = Cp[c, pad + s]
                vr = val[c]
                for s in range(n):
                    vr[s] = NEG
                oi = 0
                while oi + 4 <= no:
                    o0, o1, o2, o3 = offs[kd, oi], offs[kd, oi + 1], offs[kd, oi + 2], offs[kd, oi + 3]
                    p0, p1, p2, p3 = Pw[kd, oi], Pw[kd, oi + 1], Pw[kd, oi + 2], Pw[kd, oi + 3]
                    c0 = Cp[c, pad + o0:pad + o0 + n]
                    c1 = Cp[c, pad + o1:pad + o1 + n]
                    c2 = Cp[c, pad + o2:pad + o2 + n]
                    c3 = Cp[c, pad + o3:pad + o3 + n]
                    for s in range(n):
                        a = c0[s] - p0[s]
                        b = c1[s] - p1[s]
                        a = a if a > b else b
                        b = c2[s] - p2[s]
                        d = c3[s] - p3[s]
                        b = b if b > d else d
                        a = a if a > b else b
                        vr[s] = a if a > vr[s] else vr[s]
                    oi += 4
                while oi < no:
                    o = offs[kd, oi]
                    pw = Pw[kd, oi]
                    cs = Cp[c, pad + o:pad + o + n]
                    for s in range(n):
                        a = cs[s] - pw[s]
                        vr[s] = a if a > vr[s] else vr[s]
                    oi += 1
            if leave[t]:                           # side at t opposite to that at t1
                for s in range(n):
                    cc = 1 if side[s] > 0 else 0
                    Cp[0, pad + s] = E[t, s] + val[cc, s]
                nc = 1
            else:
                for c in range(nc):
                    for s in range(n):
                        Cp[c, pad + s] = E[t, s] + val[c, s]
        best, c, k = Cp[0, pad], 0, 0
        for cc in range(nc):
            for s in range(n):
                if Cp[cc, pad + s] > best:
                    best, c, k = Cp[cc, pad + s], cc, s
        # traceback (the same (copy, state) walk as chirplet_ridge._viterbi_flips)
        path[T - 1] = k
        for t in range(T - 1, 0, -1):
            if leave[t]:
                c = 1 if side[k] > 0 else 0
            kd = kind[t]
            vb, arg = NEG, 0
            for oi in range(nof[kd]):
                j = k + offs[kd, oi]
                if j >= 0 and j < n:
                    v = Cu[t, c, j] - Pw[kd, oi, k]
                    if v > vb:
                        vb, arg = v, j
            if enter[t] and not leave[t]:
                c = 0
            k = arg
            path[t - 1] = k
        return best

    @_jit
    def _fine_pen(dk, db, free, sh_t, sh_p, lam_k, lam, max_dk, max_db):
        """ridge_dp's pen of one fine move (inf: not allowed), as the original:
        lam_k (dk - (sh_t + sh_p) / 2) ** 2 + lam db ** 2."""
        if abs(dk) > max_dk or ((not free) and abs(db) > max_db):
            return np.inf
        tmp = np.float64(dk) - 0.5 * (sh_t + sh_p)
        return lam_k * (tmp * tmp) + lam * np.float64(db * db)

    @_jit
    def _dp_fine(E, kind, enter, leave, side, k_cent, b_cent, shift, r_k, r_b, B,
                 lam_k, lam_b, lam_b_cross, max_dk, max_db, path, C, val, Cu, Q):
        """chirplet_ridge._viterbi_flips / _viterbi on the fine states (k_cent[t] +
        LK, b_cent[t] + LB), dense over the predecessors as the original. The
        penalty of j -> s, i.e. ridge_dp's pen (+ side masks), only depends on
        (LK_s - LK_j, LB_s, LB_j) within a move: it is tabulated per move in Q, laid
        out so that the penalties of a predecessor j to all states are contiguous.
        Forward pass: max scores only (4 predecessors per sweep); the traceback
        recomputes the first maximizing predecessor of the state on the path
        (increasing j, as a dense argmax) from Cu[t, copy], the scores the move
        t-1 -> t started from.
        Work arrays: C, val (2, n), Cu (T, 2, n), Q ((2 r_b + 1)^2 (4 r_k + 1))."""
        T, n = E.shape
        nb2 = 2 * r_b + 1
        na = 2 * r_k + 1
        nd = 4 * r_k + 1
        NEG = -np.inf
        C[:, :] = NEG
        SHt, SHp = np.empty(nb2), np.empty(nb2)
        jrow = np.empty(n, np.int64)               # LK index of state j (j // nb2)
        jbase = np.empty(n, np.int64)              # start of j's penalties in Q
        modej = np.empty(n, np.int64)              # 1: all states, 0: none, 2: per row
        for a in range(na):
            for q in range(nb2):
                jrow[a * nb2 + q] = a
                jbase[a * nb2 + q] = (q * nd + 2 * r_k - a) * nb2
        for s in range(n):
            C[0, s] = E[0, s]
        nc = 1
        for t in range(1, T):
            kd = kind[t]
            if enter[t]:
                for s in range(n):
                    v = C[0, s]
                    C[0, s] = v if side[t - 1, s] == 1 else NEG
                    C[1, s] = v if side[t - 1, s] == -1 else NEG
                nc = 2
            free = kd == _FLIP or kd == _FREE
            lam = lam_b_cross if free else lam_b
            dK = k_cent[t] - k_cent[t - 1]
            dB = b_cent[t] - b_cent[t - 1]
            for q in range(nb2):
                SHt[q] = shift[min(max(b_cent[t] + q - r_b, 0), B - 1)]
                SHp[q] = shift[min(max(b_cent[t - 1] + q - r_b, 0), B - 1)]
            for cj in range(nb2):
                for d in range(nd):
                    for cs in range(nb2):
                        Q[(cj * nd + d) * nb2 + cs] = _fine_pen(
                            dK + d - 2 * r_k, dB + cs - cj, free, SHt[cs], SHp[cj],
                            lam_k, lam, max_dk, max_db)
            # side masks (hold: same side as the predecessor, flip: opposite side)
            masked = kd == _HOLD or kd == _FLIP
            tpos = True
            tneg = True
            for a in range(na):
                sd = side[t, a * nb2]
                tpos = tpos and sd == 1
                tneg = tneg and sd == -1
            for j in range(n):
                mode = 1
                if masked:
                    sp = side[t - 1, j]
                    same_all = tpos if sp == 1 else tneg
                    opp_all = tneg if sp == 1 else tpos
                    if kd == _HOLD:
                        mode = 1 if same_all else (0 if opp_all else 2)
                    else:
                        mode = 1 if opp_all else (0 if same_all else 2)
                modej[j] = mode
            for c in range(nc):
                cu = Cu[t, c]
                for s in range(n):
                    cu[s] = C[c, s]
                vr = val[c]
                for s in range(n):
                    vr[s] = NEG
                j = 0
                while j < n:
                    if (j + 4 <= n and modej[j] == 1 and modej[j + 1] == 1
                            and modej[j + 2] == 1 and modej[j + 3] == 1):
                        x0, x1, x2, x3 = C[c, j], C[c, j + 1], C[c, j + 2], C[c, j + 3]
                        q0 = Q[jbase[j]:jbase[j] + n]
                        q1 = Q[jbase[j + 1]:jbase[j + 1] + n]
                        q2 = Q[jbase[j + 2]:jbase[j + 2] + n]
                        q3 = Q[jbase[j + 3]:jbase[j + 3] + n]
                        for s in range(n):
                            a = x0 - q0[s]
                            b = x1 - q1[s]
                            a = a if a > b else b
                            b = x2 - q2[s]
                            d = x3 - q3[s]
                            b = b if b > d else d
                            a = a if a > b else b
                            vr[s] = a if a > vr[s] else vr[s]
                        j += 4
                        continue
                    x0 = C[c, j]
                    q0 = Q[jbase[j]:jbase[j] + n]
                    if modej[j] == 1:
                        for s in range(n):
                            a = x0 - q0[s]
                            vr[s] = a if a > vr[s] else vr[s]
                    elif modej[j] == 2:
                        sp = side[t - 1, j]
                        for a_s in range(na):
                            if (side[t, a_s * nb2] == sp) != (kd == _HOLD):
                                continue
                            for s in range(a_s * nb2, a_s * nb2 + nb2):
                                a = x0 - q0[s]
                                vr[s] = a if a > vr[s] else vr[s]
                    j += 1
            if leave[t]:
                for s in range(n):
                    cc = 1 if side[t, s] > 0 else 0
                    C[0, s] = E[t, s] + val[cc, s]
                nc = 1
            else:
                for c in range(nc):
                    for s in range(n):
                        C[c, s] = E[t, s] + val[c, s]
        best, c, k = C[0, 0], 0, 0
        for cc in range(nc):
            for s in range(n):
                if C[cc, s] > best:
                    best, c, k = C[cc, s], cc, s
        # traceback (the same (copy, state) walk as chirplet_ridge._viterbi_flips)
        path[T - 1] = k
        for t in range(T - 1, 0, -1):
            if leave[t]:
                c = 1 if side[t, k] > 0 else 0
            kd = kind[t]
            free = kd == _FLIP or kd == _FREE
            lam = lam_b_cross if free else lam_b
            dK = k_cent[t] - k_cent[t - 1]
            dB = b_cent[t] - b_cent[t - 1]
            for q in range(nb2):
                SHt[q] = shift[min(max(b_cent[t] + q - r_b, 0), B - 1)]
                SHp[q] = shift[min(max(b_cent[t - 1] + q - r_b, 0), B - 1)]
            a_s = jrow[k]
            c_s = k - a_s * nb2
            vb, arg = NEG, 0
            for j in range(n):
                a_j = jrow[j]
                c_j = j - a_j * nb2
                q = _fine_pen(dK + a_s - a_j, dB + c_s - c_j, free, SHt[c_s], SHp[c_j],
                              lam_k, lam, max_dk, max_db)
                if kd == _HOLD and side[t, k] != side[t - 1, j]:
                    q = np.inf
                elif kd == _FLIP and side[t, k] == side[t - 1, j]:
                    q = np.inf
                v = Cu[t, c, j] - q
                if v > vb:
                    vb, arg = v, j
            if enter[t] and not leave[t]:
                c = 0
            k = arg
            path[t - 1] = k
        return best

    @_jit
    def _fine_states(Ew, ic, jc, r_k, r_b, K, B, side_k, k_cent, b_cent, Ef, side_f):
        """ridge_dp's fine level set-up from E on the (3 r_k, 3 r_b) window around
        each coarse path block: the argmax inside the block (first maximum, as
        np.argmax) gives k_cent, b_cent; the fine states k_cent + LK, b_cent + LB
        get E (-inf outside the grid) and the side of their omega bin."""
        T = Ew.shape[0]
        nb2 = 2 * r_b + 1
        for t in range(T):
            best = Ew[t, r_k, r_b]
            si, sj = 0, 0
            for i in range(r_k):
                for j in range(r_b):
                    v = Ew[t, r_k + i, r_b + j]
                    if v > best:
                        best, si, sj = v, i, j
            kc = min(max(ic[t] * r_k + si, 0), K - 1)
            bc = min(max(jc[t] * r_b + sj, 0), B - 1)
            k_cent[t], b_cent[t] = kc, bc
            for a in range(2 * r_k + 1):
                ks = kc + a - r_k
                for c in range(nb2):
                    bs = bc + c - r_b
                    s = a * nb2 + c
                    ok = ks >= 0 and ks < K and bs >= 0 and bs < B
                    Ef[t, s] = Ew[t, si + a, sj + c] if ok else -np.inf
                    side_f[t, s] = side_k[min(max(ks, 0), K - 1)]

    @_jit
    def _energy_window(crow, ccol, cR, foff, k_start, b_start, nk, nb, mk, mb, K, B, wk, wb,
                       out):
        """blur(R)[t, k_start[t] + [0, nk), b_start[t] + [0, nb)] (exact on the in-grid
        cells): R on the window plus the blur radius from the frame's cells, the
        grid's 'nearest' boundary reproduced by repeating its edge cells."""
        T = k_start.size
        nr, ncol = nk + 2 * mk, nb + 2 * mb
        loc = np.zeros(nr * ncol)
        patch = np.empty((nr, ncol))               # float32 values
        act = np.ones(nr, np.int8)
        o1, o2 = np.empty((nr, ncol)), np.empty((nr, ncol))
        for t in range(T):
            k0, b0 = k_start[t] - mk, b_start[t] - mb
            for i in range(foff[t], foff[t + 1]):
                dr, dc = crow[i] - k0, ccol[i] - b0
                if dr >= 0 and dr < nr and dc >= 0 and dc < ncol:
                    loc[dr * ncol + dc] = cR[i]
            for r in range(nr):
                rr = min(max(k0 + r, 0), K - 1) - k0
                for q in range(ncol):
                    qq = min(max(b0 + q, 0), B - 1) - b0
                    patch[r, q] = loc[rr * ncol + qq]
            for i in range(nr * ncol):
                loc[i] = 0.0
            _blur(patch, act, wk, wb, o1, o2)
            for r in range(nk):
                for q in range(nb):
                    out[t, r, q] = o2[mk + r, mb + q]

    @_jit
    def _readout_sums(crow, ccol, cR, cRw, foff, kp, bp, dk, db, K, R5, Ro5):
        """R and Rw_omega (float32 cell sums) on the (2 dk + 1, 2 db + 1) cells
        around the ridge (omega rows wrapped); cells without energy are 0."""
        T = kp.size
        nr, ncol = 2 * dk + 1, 2 * db + 1
        for t in range(T):
            for q in range(nr * ncol):
                R5[t, q] = 0.0
                Ro5[t, q] = 0.0
            r0, c0 = kp[t] - dk, bp[t] - db
            for i in range(foff[t], foff[t + 1]):
                dr = crow[i] - r0                  # (row - r0) % K, without a division
                if dr < 0:
                    dr += K
                    if dr < 0:
                        dr %= K
                elif dr >= K:
                    dr -= K
                    if dr >= K:
                        dr %= K
                dc = ccol[i] - c0
                if dr < nr and dc >= 0 and dc < ncol:
                    R5[t, dr * ncol + dc] = cR[i]
                    Ro5[t, dr * ncol + dc] = cRw[i]
