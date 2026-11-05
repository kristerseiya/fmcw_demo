"""
Fast path of chirplet_ridge.estimate_ifs for a single ridge on the two-sided
grid (the configuration of fmcw_aux.estimate_if_fcrm):

    t, IF = estimate_if(x, **kwargs)

gives the same values, bit for bit, as

    t, IF, _ = chirplet_ridge.estimate_ifs(x, n_components=1, mode="eq9",
                                           onesided=False, ridge="dp",
                                           prediction=True, **kwargs)
    IF = IF[0]

kwargs are those of estimate_ifs, with its defaults. Used by
fmcw_aux.estimate_if_fcrm(..., fast=True).

Why the original is slow, in particular once per window in many worker processes:
  * Memory. It builds three dense float32 FCRM volumes (T, n_omega_out,
    n_beta_out) and several float64 copies of the energy for the ridge detection
    (~1.4 GB per 18750-sample window, ~2 million page faults). Fresh memory is
    page-faulted and zeroed by the OS, and with 12 processes doing that at once a
    window took ~19 s instead of ~3.5 s.
  * Its FFTs use workers=-1: every worker process starts os.cpu_count() threads.
  * The DP builds a dense (states x states) penalty / score matrix per frame.

What this module does instead (same arithmetic, different bookkeeping):
  1. Transform + reassignment in chunks of a few frames, single-threaded in-place
     FFTs. Only the reassigned coefficients (frame, omega bin, beta bin, energy,
     omega_hat) are kept: the volumes are sums of them.
  2. Per chunk, the energy E = log(blur(R) + a) - Emax is reduced right away to
     what the coarse DP level needs: the max over each (r_k, r_b) block. log and
     the subtraction are monotone, so that is log(block max) - Emax; the few cells
     within ~1e-9 of a block max get their own log as a guard.
  3. Coarse DP: each state only looks at the window of predecessors its penalty
     allows (|dk|, |db| limits), via a strided view instead of a dense matrix.
  4. The argmax inside the blocks of the coarse path (k_cent, b_cent) and the fine
     DP's energies come from E rebuilt on a small window around that path only
     (from the kept coefficients, blur margin included). The fine penalty
     matrices depend on few distinct inputs and are cached (LRU, per process).
  5. Both DP levels compute only max scores going forward; the traceback
     recomputes the argmax of the one row per frame on the path.
  6. Read-out: R, Rw_omega rebuilt only on the cells around the ridge.
  All large arrays live in per-process work buffers that are reused from call to
  call, so after the first call there are (almost) no page faults.
Cell sums use np.bincount over the same coefficients in the same order, the blur
the same scipy.ndimage.gaussian_filter, max / argmax ties are broken the same
way, so the result matches the original exactly.

Options not covered (complex x, energy != "log", return_tfr, reassign,
nfft shorter than the window, (r_k, r_b) blocks that do not tile the grid,
with zero_crossings a coarse block straddling 0 Hz) fall back to
chirplet_ridge.estimate_ifs.

The work buffers and caches stay allocated between calls (~250 MB for an
18750-sample signal with the fmcw_aux.estimate_if_fcrm defaults);
release_buffers() frees them.
"""
from collections import OrderedDict

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided, sliding_window_view

from . import chirplet_ridge as CR

TWO_PI = 2 * np.pi
_CHUNK = 4            # frames per transform chunk (small: cache-friendly with many workers)
_GROUP = 64           # frames per group when rebuilding cells from the coefficients
_PEN_CACHE = 256      # fine DP penalty matrices kept per process (~190 kB each)

# keyword arguments of chirplet_ridge.estimate_ifs handled here (with its defaults)
_DEFAULTS = dict(sigma=10.0, hop=4, nfft=128, n_beta_in=16, beta_max_hz_s=None,
                 n_omega_out=256, n_beta_out=64, gamma=1e-2, lam_k=0.05, lam_b=0.05,
                 max_dk=None, max_db=4, r_k=8, r_b=4, remove=(6, 6), blur=(1.0, 1.0),
                 readout_cells=(2, 2), n_init=10, energy="log", zero_crossings=None,
                 hold_sign=True, lam_b_cross=0.0, fs=1.0)


def estimate_if(x, **kwargs):
    """IF of the strongest ridge, the same values as
        t, IF, _ = chirplet_ridge.estimate_ifs(x, n_components=1, mode="eq9",
                                               onesided=False, ridge="dp",
                                               prediction=True, **kwargs)
        IF = IF[0]
    kwargs: those of estimate_ifs (sigma, hop, nfft, ..., zero_crossings,
    hold_sign, lam_b_cross, fs), with its defaults.
    Returns t (T,) in units of 1 / fs and IF (T,) in units of fs."""
    p = dict(_DEFAULTS)
    p.update(kwargs)
    xa = np.asarray(x)
    if not _supported(xa, p, kwargs):
        ret = CR.estimate_ifs(x, n_components=1, mode="eq9", onesided=False,
                              ridge="dp", prediction=True, **kwargs)
        return ret[0], ret[1][0]
    fs = p["fs"]
    t, w = _estimate_if(xa, p)
    return t / fs, w * fs / TWO_PI             # as chirplet_ridge.estimate_ifs


def release_buffers():
    """Free the work buffers and caches kept between calls (they are allocated
    again by the next call)."""
    _WORK.clear()
    _SETUP.clear()


def _supported(x, p, kwargs):
    if set(kwargs) - set(_DEFAULTS):           # return_tfr, reassign, ...
        return False
    if x.ndim != 1 or not np.isrealobj(x) or x.size < 2:
        return False
    if p["energy"] != "log" or p["n_beta_in"] < 2 or p["n_beta_out"] < 2:
        return False
    if p["nfft"] < 2 * int(np.ceil(4 * p["sigma"])):
        return False
    K, B = p["n_omega_out"], p["n_beta_out"]
    r_k, r_b = p["r_k"], min(p["r_b"], B)
    if K % r_k or B % r_b:
        return False
    if p["zero_crossings"] is not None:        # no coarse block may straddle 0 Hz
        side = (-np.pi + (TWO_PI / K) * np.arange(K)) >= 0
        blk = side.reshape(K // r_k, r_k)
        if not np.all(blk == blk[:, :1]):
            return False
    return True


# =========================================================================
# per-process state: work buffers and static setup
# =========================================================================
_WORK = {}


def _buf(name, shape, dtype=np.float64):
    """Work buffer `name` viewed as `shape` (uninitialized). The memory is kept and
    reused by later calls: no page faults once it has been touched."""
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    size = int(np.prod(shape))
    b = _WORK.get(name)
    if b is None or b.dtype != np.dtype(dtype) or b.size < size:
        b = np.empty(max(1, size + size // 8), dtype)
        _WORK[name] = b
    return b[:size].reshape(shape)


_SETUP = {}


class _Setup:
    pass


def _setup(p):
    key = tuple(p[k] for k in ("sigma", "hop", "nfft", "n_beta_in", "beta_max_hz_s",
                               "n_omega_out", "n_beta_out", "fs"))
    S = _SETUP.get(key)
    if S is not None:
        return S
    S = _Setup()
    sigma, nfft, Nb, fs = p["sigma"], p["nfft"], p["n_beta_in"], p["fs"]
    beta_max = (None if p["beta_max_hz_s"] is None
                else p["beta_max_hz_s"] * TWO_PI / fs ** 2)        # estimate_ifs
    # ---- chirplet_reassign, two-sided grid (omega_range=None) ----
    L = 2 * int(np.ceil(4 * sigma))
    if beta_max is None:
        beta_max = np.pi / (4 * sigma)
    S.beta_in = np.linspace(-beta_max, beta_max, Nb)
    S.beta_out = np.linspace(-beta_max, beta_max, p["n_beta_out"])
    S.d_b = S.beta_out[1] - S.beta_out[0]
    lo, hi = -np.pi, np.pi
    S.lo, S.full_circle = lo, bool(np.isclose(hi - lo, TWO_PI))
    S.Ko, S.Bo = p["n_omega_out"], p["n_beta_out"]
    S.d_om = (hi - lo) / S.Ko
    S.omega_out = lo + S.d_om * np.arange(S.Ko)
    S.omega_in = TWO_PI * (np.arange(nfft) - nfft // 2) / nfft
    tau = np.arange(L) - L // 2
    win = np.exp(-tau ** 2 / (2 * sigma ** 2)) / np.sqrt(sigma)
    base = win[None, :] * np.exp(-0.5j * S.beta_in[:, None] * tau[None, :] ** 2)
    S.mult = np.stack([base, base * (-1j * tau), base * (-(tau ** 2))])
    S.n_neg = int(np.sum(tau < 0))             # tau % nfft: [nfft - n_neg, nfft), [0, ..)
    S.tau, S.L, S.nfft, S.Nb, S.s2 = tau, L, nfft, Nb, sigma ** 2
    S.b_step = max(S.beta_in[1] - S.beta_in[0], 1e-30)
    # ---- extract_ridges / readout ----
    S.side = np.where(S.omega_out >= 0, 1, -1)    # non-decreasing in k
    S.readout_wrap = bool(np.isclose(S.Ko * S.d_om, TWO_PI))
    S.shift = S.beta_out * p["hop"] / S.d_om       # prediction=True, ridge="dp"
    S.coarse = {}                                  # coarse DP tables
    S.fine = {}                                    # fine DP penalty caches
    _SETUP[key] = S
    return S


# =========================================================================
# main
# =========================================================================
def _estimate_if(x, p):
    S = _setup(p)
    hop = p["hop"]
    K, B = S.Ko, S.Bo
    r_k, r_b = p["r_k"], min(p["r_b"], B)
    blur = (p["blur"][0], p["blur"][1])        # B > 1 here
    n0 = np.arange(0, x.size, hop)
    T = n0.size

    # ---- 1. transform, reassignment, coarse energy ----
    coef, offs, Ec, m_fr, Emax_fr = _transform(x, n0, S, p["gamma"], blur, r_k, r_b)

    # ---- 2. extract_ridges -> ridge_dp, coarse level ----
    shift = S.shift
    max_dk = p["max_dk"]
    if max_dk is None:
        max_dk = int(np.ceil(np.abs(S.beta_out).max() * hop / S.d_om)) + 6
    max_db = min(p["max_db"], B - 1)
    lam_k, lam_b, lam_b_cross = p["lam_k"], p["lam_b"], p["lam_b_cross"]
    hold_sign = p["hold_sign"]
    flips = None
    if p["zero_crossings"] is not None:
        flips = CR._flip_frames(p["zero_crossings"], n0)

    Kc, Bc = K // r_k, B // r_b
    side_c = None
    if flips is not None:
        # ridge_dp: side of a coarse cell = side of its best fine cell, which lies in
        # the rows of its block -> constant in time, as no block straddles 0 Hz
        # (checked in _supported)
        side_c = np.repeat(S.side.reshape(Kc, r_k)[:, 0], Bc)
    tabs = _coarse_tables(S, shift, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross,
                          max_dk, max_db, side_c)
    pc = _dp(Ec, _CoarseStep(tabs, Kc, Bc),
             None if side_c is None else (lambda t: side_c), flips, hold_sign, "coarse")
    ic, jc = np.divmod(pc, Bc)

    # E on the blocks of the coarse path and their neighbours (rows (ic-1..ic+1) r_k,
    # columns (jc-1..jc+1) r_b): the block argmax (ridge_dp's arg) and the fine cells
    Ew = _energy_window(coef, offs, T, S, (ic - 1) * r_k, (jc - 1) * r_b, 3 * r_k, 3 * r_b,
                        blur, m_fr, Emax_fr)
    sub = Ew[:, r_k:2 * r_k, r_b:2 * r_b].reshape(T, -1).argmax(axis=1)
    si, sj = np.divmod(sub, r_b)
    k_cent = np.clip(ic * r_k + si, 0, K - 1)
    b_cent = np.clip(jc * r_b + sj, 0, B - 1)

    # ---- 3. fine level (states: k_cent + LKr, b_cent + LBr) ----
    LK, LB = np.meshgrid(np.arange(-r_k, r_k + 1), np.arange(-r_b, r_b + 1), indexing="ij")
    LKr, LBr = LK.ravel(), LB.ravel()
    # window row of k_cent + LK is (k_cent + LK) - (ic - 1) r_k = si + r_k + LK
    Efw = Ew[np.arange(T)[:, None, None],
             si[:, None, None] + np.arange(2 * r_k + 1)[None, :, None],
             sj[:, None, None] + np.arange(2 * r_b + 1)[None, None, :]]
    Ks, Bs = k_cent[:, None] + LKr, b_cent[:, None] + LBr
    valid = (Ks >= 0) & (Ks < K) & (Bs >= 0) & (Bs < B)
    Ef = np.where(valid, Efw.reshape(T, -1), -np.inf)
    pkey = (lam_k, lam_b, lam_b_cross, max_dk, max_db, r_k, r_b)
    cache = S.fine.get(pkey)
    if cache is None:
        cache = S.fine[pkey] = _PenCache(LKr.size, _PEN_CACHE)
    step = _FineStep(k_cent, b_cent, LKr, LBr, S, cache, flips is not None,
                     lam_k, lam_b, lam_b_cross, max_dk, max_db)
    side_at = None if flips is None else step.side
    pf = _dp(Ef, step, side_at, flips, hold_sign, "fine")
    kp, bp = k_cent + LKr[pf], b_cent + LBr[pf]

    # ---- 4. read-out ----
    w = _readout(coef, offs, T, S, kp, bp, *p["readout_cells"])
    return n0, w


# =========================================================================
# 1. transform + reassignment (chirplet_reassign, mode="eq9") + coarse energy
# =========================================================================
def _transform(x, n0, S, gamma, blur, r_k, r_b):
    """Returns
      coef  : (t, ko, kb, e, w) of every reassigned coefficient, in the order in
              which chirplet_reassign accumulates them (R = bincount of e,
              Rw_omega of e * w; summed in float64, stored as float32)
      offs  : coef index of frames 0, _GROUP, 2 _GROUP, ..., T
      Ec    : ridge_dp's coarse energy per (frame, block)
      m_fr, Emax_fr : per frame max of blur(R) and of log(...) (_energy_log)"""
    L, nfft, Nb, s2 = S.L, S.nfft, S.Nb, S.s2
    K, B = S.Ko, S.Bo
    cell = K * B
    Kc, Bc = K // r_k, B // r_b
    T = n0.size
    xp = np.pad(x.astype(complex), (L, L))
    mult, tau, nn = S.mult, S.tau, S.n_neg
    beta_in, omega_in = S.beta_in, S.omega_in
    h = nfft // 2                              # fftshift: out[h:] = in[:n-h], out[:h] = in[n-h:]
    nk = nfft * Nb

    cap = T * nk                               # at most every coefficient is kept
    ct, cko, ckb = _buf("ct", cap, np.int32), _buf("cko", cap, np.int32), _buf("ckb", cap, np.int32)
    ce, cw = _buf("ce", cap), _buf("cw", cap)
    Ec = _buf("Ec", (T, Kc * Bc))
    m_fr = np.empty(T)
    Emax_fr = np.empty(T)
    # Two work areas, each used twice per chunk, so that a worker touches as little
    # memory as possible (many workers share the L3 cache and the memory bandwidth):
    #   A: FFT buffer (complex), then blur(R) (float64)
    #   B: shifted FFT (complex), then (rarely) the log energy (float64)
    sz = _CHUNK * max(3 * Nb * nfft * 2, cell)
    A = _buf("A", sz)
    Bw = _buf("B", sz)
    buf = A[:3 * _CHUNK * Nb * nfft * 2].view(complex).reshape(3, _CHUNK, Nb, nfft)
    G = Bw[:3 * _CHUNK * Nb * nfft * 2].view(complex).reshape(3, _CHUNK, nfft, Nb)
    R32 = _buf("R32", (_CHUNK, K, B), np.float32)
    n_tot = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        for s in range(0, T, _CHUNK):
            c = min(_CHUNK, T - s)
            frames = xp[n0[s:s + c, None] + L + tau[None, :]]
            bf = buf[:, :c]
            bf[..., L - nn:nfft - nn] = 0                              # zero padding
            np.multiply(frames[None, :, None, :nn], mult[:, None, :, :nn], out=bf[..., nfft - nn:])
            np.multiply(frames[None, :, None, nn:], mult[:, None, :, nn:], out=bf[..., :L - nn])
            F = sfft.fft(bf, axis=-1, workers=1, overwrite_x=True)
            g = G[:, :c]                                               # shifted, (3, c, nfft, Nb)
            g[:, :, h:, :] = F[..., :nfft - h].transpose(0, 1, 3, 2)
            g[:, :, :h, :] = F[..., nfft - h:].transpose(0, 1, 3, 2)
            C = g[0]
            P = np.abs(C) ** 2
            keep = (P >= gamma * P.max(axis=(1, 2), keepdims=True)) & (P > 0)
            # reassignment operators on the kept coefficients only (elementwise)
            idx = np.flatnonzero(keep)                                 # C order = (t, k, b)
            Cf, Cwf, Cwwf = C.reshape(-1), g[1].reshape(-1), g[2].reshape(-1)
            ti, rem = np.divmod(idx, nk)
            ki, bi0 = np.divmod(rem, Nb)
            Cs, Cws, Cwws = Cf[idx], Cwf[idx], Cwwf[idx]
            D = Cs * Cwws - Cws ** 2
            b_hat = beta_in[bi0] + np.imag(Cs ** 2 / D)
            bi = np.clip(np.rint((b_hat - beta_in[0]) / S.b_step), 0, Nb - 1)
            bi = np.where(np.isfinite(bi), bi, 0).astype(int)
            ig = idx - bi0 + bi
            w_hat = omega_in[ki] + np.real(Cwf[ig] / Cf[ig]) / s2
            ok = np.isfinite(b_hat)
            w_hat = CR._wrap(w_hat)
            ok &= np.isfinite(w_hat)
            e, w, b, tl = P.reshape(-1)[idx][ok], w_hat[ok], b_hat[ok], ti[ok]
            # _accumulate
            ko = np.rint((w - S.lo) / S.d_om).astype(int)
            if S.full_circle:
                ko %= K
            kb = np.rint((b - S.beta_out[0]) / S.d_b).astype(int)
            ok = (ko >= 0) & (ko < K) & (kb >= 0) & (kb < B)
            tl, ko, kb, e, w = tl[ok], ko[ok], kb[ok], e[ok], w[ok]
            R = R32[:c]
            np.copyto(R.reshape(-1), np.bincount((tl * K + ko) * B + kb, e, c * cell))
            sl = slice(n_tot, n_tot + tl.size)
            ct[sl], cko[sl], ckb[sl], ce[sl], cw[sl] = tl + s, ko, kb, e, w
            n_tot += tl.size

            # _energy_log: E = log(blur(R) + a + 1e-300) - Emax, a = 1e-6 max(blur(R))
            # (no blur along time: frames are independent). ridge_dp's coarse level
            # only needs the max of E over each (r_k, r_b) block; the argmax inside
            # the block is only needed on the coarse path (later, _energy_window).
            Rb = A[:c * cell].reshape(c, K, B)
            gaussian_filter(R, (0, blur[0], blur[1]), mode="nearest", output=Rb)
            Rb4 = Rb.reshape(c, Kc, r_k, B)
            rbm = Rb4.max(axis=2).reshape(c, Kc, Bc, r_b).max(axis=-1)   # block max of blur(R)
            m = rbm.max(axis=(1, 2))
            a = 1e-6 * m
            m_fr[s:s + c] = m
            if a.min() > 1e-280:
                # v + 1e-300 == v here, and x -> fl(x + a) is monotone: the block max of
                # log(V), V = blur(R) + a, is log(block max of V). Cells whose V is
                # within ~2e-9 of it are checked explicitly (in case log is not
                # exactly monotone); everything else is far below.
                lm = np.log(rbm + a[:, None, None])
                col = np.repeat(rbm, r_b, axis=-1)[:, :, None, :]
                near = (Rb4 >= col - 2e-9 * (col + a[:, None, None, None])) & (Rb4 < col)
                if near.any():
                    tt, kk, ii, bb = np.nonzero(near)
                    np.maximum.at(lm, (tt, kk, bb // r_b), np.log(Rb4[tt, kk, ii, bb] + a[tt]))
            else:                                  # as the original: log of every cell
                X = Bw[:c * cell].reshape(c, K, B)
                np.add(Rb, a[:, None, None], out=X)
                X += 1e-300
                np.log(X, out=X)
                lm = X.reshape(c, Kc, r_k, B).max(axis=2).reshape(c, Kc, Bc, r_b).max(axis=-1)
            # x -> x - Emax is monotone too: max(E) over a block = (max log V) - Emax
            Em = lm.max(axis=(1, 2))
            Emax_fr[s:s + c] = Em
            np.subtract(lm.reshape(c, -1), Em[:, None], out=Ec[s:s + c])

    coef = (ct[:n_tot], cko[:n_tot], ckb[:n_tot], ce[:n_tot], cw[:n_tot])
    offs = np.searchsorted(coef[0], np.minimum(np.arange(0, T + _GROUP, _GROUP), T))
    return coef, offs, Ec, m_fr, Emax_fr


def _groups(coef, offs, T):
    """(first frame, number of frames, coefficients of these frames) per _GROUP frames."""
    for i, f0 in enumerate(range(0, T, _GROUP)):
        sl = slice(offs[i], offs[i + 1])
        yield f0, min(_GROUP, T - f0), tuple(a[sl] for a in coef)


# =========================================================================
# 2./3. dynamic programming (chirplet_ridge._viterbi_flips / _viterbi)
# =========================================================================
def _dp(E, step, side_at, flips, hold_sign, name):
    """chirplet_ridge._viterbi_flips (flips given) or _viterbi (flips None) on
    E_frames = E (T, n), with kind of move as in _viterbi_flips and
      step(t, kind, C) -> val[copy, state] = max_j (C[copy, j] - pen_t[state, j])
      step.pred(t, kind, C_row, state) -> first j maximizing C_row[j] - pen_t[state, j]
                                          (0 if all are -inf, as a dense argmax)
    The backpointers are not computed in the forward pass: the scores C each move
    started from are kept, and the traceback recomputes the argmax of the single
    row on the path (the same row the original takes its argmax of)."""
    T, n = E.shape
    plain = flips is None
    flips = [] if plain else flips
    inside = np.zeros(T, bool)
    free = np.zeros(T, bool)
    for t1, t2 in flips:
        inside[t1 + 1:t2] = True
        free[t1 + 1:t2 + 1] = True
    split = np.array([[1], [-1]])
    rows = np.arange(n)
    Cu = _buf("dpC_" + name, (T, 2, n))          # Cu[t]: C the move t-1 -> t started from
    mode = np.zeros(T, np.int8)                  # move t-1 -> t: 0 normal, 1 enter, 2 leave
    kinds = [None] * T
    left = {}
    C = E[0][None, :]
    for t in range(1, T):
        enter, leave = inside[t] and not inside[t - 1], inside[t - 1] and not inside[t]
        if not free[t]:
            kind = "hold" if (hold_sign and not plain) else "plain"
        elif not (inside[t] or inside[t - 1]):
            kind = "flip"
        else:
            if enter:                          # split by side at t1
                C = np.where(side_at(t - 1)[None, :] == split, C, -np.inf)
            kind = "free"
        Cu[t, :C.shape[0]] = C
        kinds[t] = kind
        val = step(t, kind, C)
        if leave:                              # side at t2 opposite to that at t1
            cc = (side_at(t) > 0).astype(int)
            left[t] = cc
            mode[t] = 2
            val = val[cc, rows][None, :]
        else:
            mode[t] = enter
        C = E[t][None, :] + val
    c, k = np.unravel_index(np.argmax(C), C.shape)
    if not plain and not np.isfinite(C[c, k]):
        raise ValueError("no ridge satisfies the zero-crossing constraints (an interval "
                         "too short for max_dk, or hold_sign with a sign change elsewhere)")
    path = np.empty(T, int)
    path[-1] = k
    for t in range(T - 1, 0, -1):              # same (copy, state) walk as the original
        if mode[t] == 2:
            c = left[t][k]
            k = step.pred(t, kinds[t], Cu[t, c], k)
        else:
            k = step.pred(t, kinds[t], Cu[t, c], k)
            if mode[t] == 1:
                c = 0
        path[t - 1] = k
    return path


def _coarse_pen(shift, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross, max_dk, max_db):
    """ridge_dp's coarse penalty matrices Pc, Pc_free."""
    B = Bc * r_b
    CK, CB = np.meshgrid(np.arange(Kc) * r_k + (r_k - 1) / 2,
                         np.arange(Bc) * r_b + (r_b - 1) / 2, indexing="ij")
    CK, CB = CK.ravel(), CB.ravel()
    sh_c = np.interp(CB, np.arange(B), shift)
    dk = CK[:, None] - CK[None, :]
    db = CB[:, None] - CB[None, :]
    Pk = lam_k * (dk - 0.5 * (sh_c[:, None] + sh_c[None, :])) ** 2
    Pc = Pk + lam_b * db ** 2
    Pc[(np.abs(dk) > max_dk + r_k) | (np.abs(db) > max_db + r_b)] = np.inf
    Pc_free = Pk + lam_b_cross * db ** 2
    Pc_free[np.abs(dk) > max_dk + r_k] = np.inf
    return Pc, Pc_free


def _coarse_tables(S, shift, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross, max_dk, max_db,
                   side):
    """Per kind of move: the penalties of the window of allowed predecessors of every
    coarse state (cached per process: they only depend on the parameters)."""
    key = (Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross, max_dk, max_db,
           None if side is None else side.tobytes())
    tabs = S.coarse.get(key)
    if tabs is not None:
        return tabs
    Pc, Pc_free = _coarse_pen(shift, Kc, Bc, r_k, r_b, lam_k, lam_b, lam_b_cross,
                              max_dk, max_db)
    n = Kc * Bc
    if side is None:
        kinds = {"plain": Pc}
    else:
        same = side[:, None] == side[None, :]
        kinds = {"hold": np.where(same, Pc, np.inf), "plain": Pc,
                 "flip": np.where(same, np.inf, Pc_free), "free": Pc_free}
    ic, jc = np.divmod(np.arange(n), Bc)
    s = np.arange(n)
    tabs = {}
    for kind, P in kinds.items():
        r, q = np.nonzero(np.isfinite(P))
        U = int(np.abs(q // Bc - r // Bc).max()) if r.size else 0
        V = int(np.abs(q % Bc - r % Bc).max()) if r.size else 0
        u, v = np.arange(-U, U + 1), np.arange(-V, V + 1)
        nu, nv = u.size, v.size
        if Bc > 2 * V:
            # flat layout: predecessor of state s at offset (du, dv) is s + du Bc + dv,
            # strictly increasing in (du, dv) since Bc > 2 V; offsets that leave the
            # grid or wrap to another row get an infinite penalty
            off = (u[:, None] * Bc + v[None, :]).ravel()
            pic = ic[None, :] + np.repeat(u, nv)[:, None]               # (nu nv, n)
            pjc = jc[None, :] + np.tile(v, nu)[:, None]
            inr = (pic >= 0) & (pic < Kc) & (pjc >= 0) & (pjc < Bc)
            Pw = np.where(inr, P[s[None, :], np.where(inr, s[None, :] + off[:, None], 0)], np.inf)
            pad0 = U * Bc + V
            pad = np.full((2, n + 2 * pad0), -np.inf)
            it = pad.itemsize
            tabs[kind] = dict(flat=True, pad0=pad0, Pw=Pw.reshape(nu, nv, n), pad=pad,
                              win=as_strided(pad, (2, nu, nv, n), (pad.strides[0], Bc * it, it, it),
                                             writeable=False),
                              M=np.empty((2, nu, nv, n)),
                              pred=np.clip(s[:, None] + off[None, :], 0, n - 1),
                              Pwp=np.ascontiguousarray(Pw.T))
        else:
            # 2-D layout (predecessor order: (ic + du, jc + dv) lexicographic)
            pic = ic[:, None, None] + u[None, :, None]
            pjc = jc[:, None, None] + v[None, None, :]
            inr = (pic >= 0) & (pic < Kc) & (pjc >= 0) & (pjc < Bc)
            pidx = np.where(inr, pic * Bc + pjc, 0)                       # (n, nu, nv)
            Pw = np.where(inr, P[s[:, None, None], pidx], 0.0)             # outside: C = -inf
            pad = np.full((2, Kc + 2 * U, Bc + 2 * V), -np.inf)
            tabs[kind] = dict(flat=False, U=U, V=V, Pw=Pw.reshape(Kc, Bc, nu, nv), pad=pad,
                              win=sliding_window_view(pad, (nu, nv), axis=(1, 2)),
                              M=np.empty((2, Kc, Bc, nu, nv)),
                              pred=pidx.reshape(n, -1),
                              Pwp=np.where(inr, Pw, np.inf).reshape(n, -1))
        # pred[s], Pwp[s]: allowed predecessors of s in increasing order and their
        # penalties (inf: not a predecessor), for the traceback
    S.coarse[key] = tabs
    return tabs


class _CoarseStep:
    """step() of _dp for the coarse level: max over the window of allowed
    predecessors (the others are -inf in the dense original, so the max and the
    first maximizing predecessor are the same)."""

    def __init__(self, tabs, Kc, Bc):
        self.tabs, self.Kc, self.Bc, self.n = tabs, Kc, Bc, Kc * Bc

    def __call__(self, t, kind, C):
        tb = self.tabs[kind]
        n, cp = self.n, C.shape[0]
        M = tb["M"][:cp]
        if tb["flat"]:
            p0 = tb["pad0"]
            tb["pad"][:cp, p0:p0 + n] = C
            np.subtract(tb["win"][:cp], tb["Pw"], out=M)          # (cp, nu, nv, n)
            return M.reshape(cp, -1, n).max(axis=1)
        U, V, Kc, Bc = tb["U"], tb["V"], self.Kc, self.Bc
        tb["pad"][:cp, U:U + Kc, V:V + Bc] = C.reshape(cp, Kc, Bc)
        np.subtract(tb["win"][:cp], tb["Pw"], out=M)
        return M.reshape(cp, n, -1).max(axis=-1)

    def pred(self, t, kind, Crow, k):
        tb = self.tabs[kind]
        p = tb["pred"][k]
        v = Crow[p] - tb["Pwp"][k]
        i = int(np.argmax(v))                  # first max = smallest predecessor
        return int(p[i]) if v[i] > -np.inf else 0


class _PenCache:
    """LRU cache of (n, n) penalty matrices in one preallocated pool."""

    def __init__(self, n, cap):
        self.pool = np.empty((cap, n, n))
        self.slots = OrderedDict()
        self.free = list(range(cap))

    def get(self, key, compute):
        slot = self.slots.get(key)
        if slot is not None:
            self.slots.move_to_end(key)
            return self.pool[slot]
        slot = self.free.pop() if self.free else self.slots.popitem(last=False)[1]
        self.pool[slot] = compute()
        self.slots[key] = slot
        return self.pool[slot]


class _FineStep:
    """step() of _dp for the fine level: dense, as the original. The penalty of the
    move t-1 -> t only depends on (k_cent[t] - k_cent[t-1], b_cent[t], b_cent[t-1])
    and, for the side masks, on the side patterns of the two frames: cached."""

    def __init__(self, k_cent, b_cent, LKr, LBr, S, cache, sides, lam_k, lam_b,
                 lam_b_cross, max_dk, max_db):
        self.k_cent, self.b_cent, self.LKr, self.LBr = k_cent, b_cent, LKr, LBr
        self.K, self.B, self.side_k, self.shift = S.Ko, S.Bo, S.side, S.shift
        self.cache = cache
        self.lam_k, self.lam_b, self.lam_b_cross = lam_k, lam_b, lam_b_cross
        self.max_dk, self.max_db = max_dk, max_db
        self.dK = np.diff(k_cent, prepend=k_cent[0])
        if sides:                              # side pattern id per frame (side is monotone)
            lo = S.side[np.clip(k_cent + LKr.min(), 0, S.Ko - 1)]
            hi = S.side[np.clip(k_cent + LKr.max(), 0, S.Ko - 1)]
            self.sid = np.where(lo > 0, -1, np.where(hi < 0, -2, k_cent))

    def side(self, t):                         # side[Kcl[t]]
        return self.side_k[np.clip(self.k_cent[t] + self.LKr, 0, self.K - 1)]

    def pen(self, t, free=False):              # chirplet_ridge.ridge_dp's pen
        Ks_t, Ks_p = self.k_cent[t] + self.LKr, self.k_cent[t - 1] + self.LKr
        Bs_t, Bs_p = self.b_cent[t] + self.LBr, self.b_cent[t - 1] + self.LBr
        sh_t = self.shift[np.clip(Bs_t, 0, self.B - 1)]
        sh_p = self.shift[np.clip(Bs_p, 0, self.B - 1)]
        dk = Ks_t[:, None] - Ks_p[None, :]
        P = self.lam_k * (dk - 0.5 * (sh_t[:, None] + sh_p[None, :])) ** 2
        db = Bs_t[:, None] - Bs_p[None, :]
        P += (self.lam_b_cross if free else self.lam_b) * db ** 2
        bad = np.abs(dk) > self.max_dk
        if not free:
            bad |= np.abs(db) > self.max_db
        P[bad] = np.inf
        return P

    def _masked(self, t, kind):
        P = self.pen(t, kind in ("flip", "free"))
        if kind in ("hold", "flip"):
            same = self.side(t)[:, None] == self.side(t - 1)[None, :]
            P = np.where(same, P, np.inf) if kind == "hold" else np.where(same, np.inf, P)
        return P

    def _pen(self, t, kind):
        if kind in ("hold", "flip"):
            key = (kind, self.dK[t], self.b_cent[t], self.b_cent[t - 1],
                   self.sid[t], self.sid[t - 1])
        else:
            key = (kind, self.dK[t], self.b_cent[t], self.b_cent[t - 1])
        return self.cache.get(key, lambda: self._masked(t, kind))

    def __call__(self, t, kind, C):
        return (C[:, None, :] - self._pen(t, kind)[None]).max(axis=-1)

    def pred(self, t, kind, Crow, k):
        return int(np.argmax(Crow - self._pen(t, kind)[k]))


# =========================================================================
# 3. energy on the fine cells, 4. read-out
# =========================================================================
def _local_sums(f0, nf, cf, rows0, nr, cols0, ncol, K, with_w, wrap_rows=False):
    """float32 cell sums of e (and e * w) as in chirplet_reassign, for frames
    f0 .. f0 + nf - 1 (coefficients cf), on the window rows0 + [0, nr),
    cols0 + [0, ncol) of each frame (rows0, cols0 per frame); same coefficients in
    the same order as the full sum. Cells outside the grid stay 0."""
    t, ko, kb, e, w = cf
    tl = t - f0
    dr = ko - rows0[tl]
    if wrap_rows:
        dr %= K
    dc = kb - cols0[tl]
    sel = (dr >= 0) & (dr < nr) & (dc >= 0) & (dc < ncol)
    loc = (tl[sel] * nr + dr[sel]) * ncol + dc[sel]
    es = e[sel]
    size = nf * nr * ncol
    out = [np.bincount(loc, es, size).astype(np.float32).reshape(nf, nr, ncol)]
    if with_w:
        out.append(np.bincount(loc, es * w[sel], size).astype(np.float32).reshape(nf, nr, ncol))
    return out


def _energy_window(coef, offs, T, S, k_start, b_start, nk, nb, blur, m_fr, Emax_fr):
    """_energy_log(R, blur)[t, k_start[t] + [0, nk), b_start[t] + [0, nb)] for every
    frame t, exact on the in-grid cells (others: arbitrary values). R is rebuilt on
    the window plus the blur radius; the full grid's 'nearest' boundary is
    reproduced by replicating its edge rows / columns."""
    K, B = S.Ko, S.Bo
    mk = int(4.0 * float(blur[0]) + 0.5) if blur[0] > 1e-15 else 0   # gaussian radius
    mb = int(4.0 * float(blur[1]) + 0.5) if blur[1] > 1e-15 else 0
    nr, ncol = nk + 2 * mk, nb + 2 * mb
    out = _buf("Ewin", (T, nk, nb))
    for f0, nf, cf in _groups(coef, offs, T):
        sl = slice(f0, f0 + nf)
        k0, b0 = k_start[sl] - mk, b_start[sl] - mb
        (Rl,) = _local_sums(f0, nf, cf, k0, nr, b0, ncol, K, with_w=False)
        rows = np.clip(k0[:, None] + np.arange(nr), 0, K - 1) - k0[:, None]
        cols = np.clip(b0[:, None] + np.arange(ncol), 0, B - 1) - b0[:, None]
        patch = Rl[np.arange(nf)[:, None, None], rows[:, :, None], cols[:, None, :]]
        Rb = gaussian_filter(patch, (0, blur[0], blur[1]), mode="nearest", output=np.float64)
        Rb = Rb[:, mk:mk + nk, mb:mb + nb]
        out[sl] = np.log(Rb + 1e-6 * m_fr[sl, None, None] + 1e-300) - Emax_fr[sl, None, None]
    return out


def _readout(coef, offs, T, S, kp, bp, dk=2, db=2):
    """chirplet_ridge.readout, IF part (rad/sample)."""
    K, B = S.Ko, S.Bo
    if not S.readout_wrap:
        raise NotImplementedError("one-sided grid")
    nr, ncol = 2 * dk + 1, 2 * db + 1
    e = np.empty(T, np.float32)
    o = np.empty(T, np.float32)
    for f0, nf, cf in _groups(coef, offs, T):
        sl = slice(f0, f0 + nf)
        R, Ro = _local_sums(f0, nf, cf, kp[sl] - dk, nr, bp[sl] - db, ncol, K,
                            with_w=True, wrap_rows=True)
        full = (bp[sl] - db >= 0) & (bp[sl] + db + 1 <= B)
        e[sl][full] = R[full].reshape(-1, nr * ncol).sum(axis=1)
        o[sl][full] = Ro[full].reshape(-1, nr * ncol).sum(axis=1)
        for i in np.nonzero(~full)[0]:         # ridge at the edge of the beta grid
            t = f0 + i
            j = np.arange(max(0, bp[t] - db), min(B, bp[t] + db + 1)) - (bp[t] - db)
            ix = np.ix_(np.arange(nr), j)
            e[t] = R[i][ix].sum()
            o[t] = Ro[i][ix].sum()
    w = np.empty(T)
    pos = e > 0
    w[pos] = o[pos] / e[pos]
    w[~pos] = S.omega_out[kp[~pos]]
    return w
