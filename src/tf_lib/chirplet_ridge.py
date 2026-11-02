"""
Frequency-chirprate reassignment (FCRM, Zhu et al., "Frequency-Chirprate
Reassignment") with 3-D ridge detection.

Units: omega in rad/sample, beta (chirp rate) in rad/sample^2, sigma in
samples. The end-to-end estimators (estimate_ifs, estimate_if_chirplet) take
the sampling rate as an optional keyword fs (default 1): times are returned in
units of 1 / fs, IF in units of fs and chirp rate in units of fs^2, i.e.
samples, cycles/sample and cycles/sample^2 by default (s, Hz, Hz/s when the
true fs is given). All other parameters are in samples / bins, independent
of fs.

Chirplet transform (paper eq. 2, Gaussian window):
    C(t, w, b) = sum_tau x[t + tau] g(tau) exp(-j b tau^2 / 2) exp(-j w tau)
    C_w  = -j C^{tg},   C_ww = -C^{t^2 g}                      (eqs. 11-12)

Reassignment operators (D = C C_ww - C_w^2):
    b_hat = b + Im(C^2 / D)                                    (eq. 8 / 13)
    omega modes:
      "eq9"   w_hat = w + Re(C_w / C)|_{b = b_hat} / sigma^2   (eq. 9; the
              transform is read at the transform-grid chirp rate nearest to
              b_hat, so accuracy depends on the chirp-rate grid spacing.
              Least noisy when that grid is fine relative to the chirp-rate
              range, e.g. the paper's test signal)
      "eq6"   w_hat = w + (1 + sigma^4 (b - b_hat)^2) / sigma^2 Re(C_w / C)
              (eq. 6 with phi'' replaced by b_hat; exact for linear chirps,
              uses only the coefficients of the original cell)
      "joint" w_hat = w - Re(C C_w / D)  (equivalent closed form of eq. 6;
              robust default when the chirp-rate grid is coarse / wide)
      "two_step"  grouped-sum variant of eq. 9 from the original code
              (sums C, C_w over all cells sharing a b_hat bin; biased)
      "fsst"  w_hat = w + Re(C_w / C) / sigma^2 at b = 0 only, b_hat = 0:
              Gaussian-window Fourier synchrosqueezing (2-D baseline)
FCRM (eq. 10): |C|^2 is moved to (w_hat, b_hat).

Ridge detection on the (T, n_omega, n_beta) FCRM volume:
  ridge="greedy"  paper Algorithm 1 (greedy tracking from a start frame,
                  several starts, keep the best value of cost (14))
  ridge="dp"      exact maximizer of cost (14) by coarse-to-fine dynamic
                  programming; optional constant-chirp-rate prediction
                  (penalize the frequency move minus beta * hop instead of
                  the raw move)
Multiple components: peeling as in Algorithm 1 step 12.
Zero-crossing hints (ridge="dp", two-sided grid): sample intervals
  [idx1, idx2] each bracketing one sign change of the IF. The ridge must be on
  opposite sides of 0 Hz at idx1 and idx2, the chirp rate is unconstrained in
  between (no max_db limit; lam_b replaced by lam_b_cross, default 0), and
  with hold_sign the side stays fixed outside the intervals. Exact: the
  non-local constraint is made local by doubling the DP states inside each
  interval (one copy per side at idx1).
"""
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter

TWO_PI = 2 * np.pi


# =========================================================================
# Transform + reassignment
# =========================================================================
def chirplet_reassign(x, sigma=10.0, window_length=None, hop=4, nfft=128,
                      n_beta_in=16, beta_max=None, n_omega_out=256,
                      n_beta_out=64, mode="eq9", gamma=1e-3, chunk=64,
                      omega_range=None, return_raw=False, return_C=False,
                      return_complex=False, workers=-1, compute_R=True):
    """
    compute_R=False skips the reassignment (no R / Rw_omega / Rw_beta in the
    output): only the transform itself is computed, e.g. with return_complex=True
    for the STFT or chirplet transform alone (one FFT instead of three per frame
    unless return_raw is also requested).
    Returns dict with
      t (frame centres, samples), omega_in, beta_in, omega_out, beta_out,
      R        FCRM energy, shape (T, n_omega_out, n_beta_out)
      Rw_omega energy * omega_hat summed per output cell (for read-out)
      Rw_beta  energy * beta_hat  summed per output cell
      C2       |C|^2 on the input grid (T, nfft, n_beta_in)  [return_C]
      C        complex coefficients on the input grid (T, nfft, n_beta_in),
               complex64  [return_complex]. Window g(tau) = exp(-tau^2 /
               2 sigma^2) / sqrt(sigma); phase referenced to the frame centre
               (tau = 0 at sample t[i]); frequency axis omega_in (FFT-shifted).
               mode="fsst": n_beta_in = 1, i.e. C[:, :, 0] is the STFT.
      omega_hat, beta_hat  raw operators on the input grid  [return_raw]
    omega_range: (lo, hi) of the output frequency grid in rad/sample,
      default (-pi, pi) (wraps); use (0, pi) for real-valued signals.
    """
    x = np.asarray(x).astype(complex)
    if window_length is None:
        window_length = 2 * int(np.ceil(4 * sigma))
    L = int(window_length)
    if nfft < L:
        raise ValueError("nfft must be >= window_length")
    if beta_max is None:
        beta_max = np.pi / (4 * sigma)
    if mode == "fsst":
        n_beta_in, n_beta_out = 1, 1
        beta_in = np.zeros(1)
        beta_out = np.zeros(1)
        d_b = 1.0
    else:
        beta_in = np.linspace(-beta_max, beta_max, n_beta_in)
        beta_out = np.linspace(-beta_max, beta_max, n_beta_out)
        d_b = beta_out[1] - beta_out[0]
    lo, hi = (-np.pi, np.pi) if omega_range is None else omega_range
    full_circle = np.isclose(hi - lo, TWO_PI)
    d_om = (hi - lo) / n_omega_out
    omega_out = lo + d_om * np.arange(n_omega_out)
    omega_in = TWO_PI * (np.arange(nfft) - nfft // 2) / nfft
    grid = dict(lo=lo, d_om=d_om, Ko=n_omega_out, wrap=full_circle,
                bmin=beta_out[0], d_b=d_b, Bo=n_beta_out)

    tau = np.arange(L) - L // 2
    win = np.exp(-tau ** 2 / (2 * sigma ** 2)) / np.sqrt(sigma)
    base = win[None, :] * np.exp(-0.5j * beta_in[:, None] * tau[None, :] ** 2)
    need_ops = compute_R or return_raw                 # reassignment operators needed?
    mult = (np.stack([base, base * (-1j * tau), base * (-(tau ** 2))])   # g, C_w, C_ww
            if need_ops else base[None])
    pos = tau % nfft
    s2 = sigma ** 2

    n0 = np.arange(0, x.size, hop)
    T = n0.size
    xp = np.pad(x, (L, L))
    cell = n_omega_out * n_beta_out
    n_vol = T * cell if compute_R else 0
    R = np.zeros(n_vol, np.float32)             # output volumes (float32 to bound
    Rwo = np.zeros(n_vol, np.float32)           # memory); accumulated chunk by
    Rwb = np.zeros(n_vol, np.float32)           # chunk in float64
    C2_all = np.zeros((T, nfft, n_beta_in), np.float32) if return_C else None
    C_all = np.zeros((T, nfft, n_beta_in), np.complex64) if return_complex else None
    raw_o = np.zeros((T, nfft, n_beta_in), np.float32) if return_raw else None
    raw_b = np.zeros((T, nfft, n_beta_in), np.float32) if return_raw else None

    for s in range(0, T, chunk):
        idx = np.arange(s, min(T, s + chunk))
        c = idx.size
        frames = xp[n0[idx, None] + L + tau[None, :]]
        buf = np.zeros((mult.shape[0], c, n_beta_in, nfft), complex)
        buf[..., pos] = frames[None, :, None, :] * mult[:, None, :, :]
        F = sfft.fftshift(sfft.fft(buf, axis=-1, workers=workers), axes=-1)
        C = np.swapaxes(F[0], 1, 2)                                        # (c, nfft, Nb)
        P = np.abs(C) ** 2
        if return_C:
            C2_all[idx] = P
        if return_complex:
            C_all[idx] = C
        if not need_ops:
            continue
        Cw, Cww = np.swapaxes(F[1], 1, 2), np.swapaxes(F[2], 1, 2)
        keep = (P >= gamma * P.max(axis=(1, 2), keepdims=True)) & (P > 0)
        W = omega_in[None, :, None]
        Bg = beta_in[None, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "fsst":
                b_hat = np.zeros(C.shape)
                w_hat = W + np.real(Cw / C) / s2
            else:
                D = C * Cww - Cw ** 2
                b_hat = Bg + np.imag(C ** 2 / D)
                if mode == "joint":
                    w_hat = W - np.real(C * Cw / D)
                elif mode == "eq6":
                    w_hat = W + (1 + s2 ** 2 * (Bg - b_hat) ** 2) / s2 * np.real(Cw / C)
                elif mode == "eq9":
                    bi = np.clip(np.rint((b_hat - beta_in[0]) /
                                         max(beta_in[1] - beta_in[0], 1e-30)), 0, n_beta_in - 1)
                    bi = np.where(np.isfinite(bi), bi, 0).astype(int)
                    Cg = np.take_along_axis(C, bi, axis=2)
                    Cwg = np.take_along_axis(Cw, bi, axis=2)
                    w_hat = W + np.real(Cwg / Cg) / s2
                elif mode == "two_step":
                    w_hat = None
                else:
                    raise ValueError(mode)
        keep &= np.isfinite(b_hat)
        if w_hat is not None:
            w_hat = _wrap(w_hat)
            keep &= np.isfinite(w_hat)
        if return_raw and w_hat is not None:
            raw_b[idx], raw_o[idx] = b_hat, w_hat
        tt = np.broadcast_to((idx - s)[:, None, None], C.shape)
        views = (R[s * cell:(s + c) * cell], Rwo[s * cell:(s + c) * cell],
                 Rwb[s * cell:(s + c) * cell])

        if not compute_R:
            continue
        if mode != "two_step":
            _accumulate(views, tt[keep], w_hat[keep], b_hat[keep], P[keep], grid)
        else:
            kb = np.rint((b_hat - beta_out[0]) / d_b)
            keep &= (kb >= 0) & (kb < n_beta_out)
            kk = np.broadcast_to(np.arange(nfft)[None, :, None], C.shape)
            lin = (tt[keep] * nfft + kk[keep]) * n_beta_out + kb[keep].astype(int)
            sz = c * nfft * n_beta_out
            sC = _cbincount(lin, C[keep], sz)
            sCw = _cbincount(lin, Cw[keep], sz)
            sP = np.bincount(lin, P[keep], sz)
            sPb = np.bincount(lin, P[keep] * b_hat[keep], sz)
            nz = np.nonzero(np.abs(sC) > 0)[0]
            t_c, rem = np.divmod(nz, nfft * n_beta_out)
            k_c, _ = np.divmod(rem, n_beta_out)
            w2 = _wrap(omega_in[k_c] + np.real(sCw[nz] / sC[nz]) / s2)
            _accumulate(views, t_c, w2, sPb[nz] / sP[nz], np.abs(sC[nz]) ** 2, grid)

    shp = (T, n_omega_out, n_beta_out)
    out = dict(t=n0, omega_in=omega_in, beta_in=beta_in, omega_out=omega_out,
               beta_out=beta_out, sigma=sigma, hop=hop, d_om=d_om)
    if compute_R:
        out.update(R=R.reshape(shp), Rw_omega=Rwo.reshape(shp), Rw_beta=Rwb.reshape(shp))
    if return_C:
        out["C2"] = C2_all
    if return_complex:
        out["C"] = C_all
    if return_raw:
        out["omega_hat"], out["beta_hat"] = raw_o, raw_b
    return out


def _wrap(w):
    return (w + np.pi) % TWO_PI - np.pi


def _cbincount(lin, z, size):
    return np.bincount(lin, z.real, size) + 1j * np.bincount(lin, z.imag, size)


def _accumulate(views, t_loc, w, b, e, g):
    """Add energy e at (chunk-local frame, nearest omega bin, nearest beta bin)
    into the chunk's slices of the output volumes."""
    R, Rwo, Rwb = views
    ko = np.rint((w - g["lo"]) / g["d_om"]).astype(int)
    if g["wrap"]:
        ko %= g["Ko"]
    kb = np.rint((b - g["bmin"]) / g["d_b"]).astype(int) if g["Bo"] > 1 else np.zeros(b.size, int)
    ok = (ko >= 0) & (ko < g["Ko"]) & (kb >= 0) & (kb < g["Bo"])
    lin = (t_loc[ok] * g["Ko"] + ko[ok]) * g["Bo"] + kb[ok]
    n = R.size
    R += np.bincount(lin, e[ok], n).astype(np.float32)
    Rwo += np.bincount(lin, e[ok] * w[ok], n).astype(np.float32)
    Rwb += np.bincount(lin, e[ok] * b[ok], n).astype(np.float32)


# =========================================================================
# Ridge detection
# =========================================================================
def _viterbi(E_frames, pen_fn):
    T = len(E_frames)
    C = E_frames[0].copy()
    back = []
    for t in range(1, T):
        M = C[None, :] - pen_fn(t)
        a = M.argmax(1)
        back.append(a)
        C = E_frames[t] + M[np.arange(a.size), a]
    path = np.empty(T, int)
    path[-1] = int(np.argmax(C))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t - 1][path[t]]
    return path


def _viterbi_flips(E_frames, pen_fn, side, flips, hold_sign=True, static=False):
    """_viterbi with sign-change constraints. flips: sorted frame pairs
    (t1, t2), t1 < t2, intervals sharing at most an end frame; side[t]: +1 / -1
    per state of frame t. The path's side at t2 must be opposite to its side
    at t1; hold_sign: no change of side outside the intervals.
    pen_fn(t, free) -> (n_t, n_{t-1}) penalty of the move t-1 -> t, free=True
    inside an interval. static: pen_fn and side do not depend on t (the masked
    penalties are then built once).
    Inside an interval the DP keeps two copies of the states, copy 0 for paths
    on side +1 at t1 and copy 1 for side -1, so the constraint is local."""
    T = len(E_frames)
    inside = np.zeros(T, bool)                 # frame strictly inside an interval
    free = np.zeros(T, bool)                   # move t-1 -> t inside an interval
    for t1, t2 in flips:
        inside[t1 + 1:t2] = True
        free[t1 + 1:t2 + 1] = True
    cache = {}

    def pen(t, kind):
        if kind in cache:
            return cache[kind]
        P = pen_fn(t, free[t])
        if kind != "free":
            same = side[t][:, None] == side[t - 1][None, :]
            P = np.where(same, P, np.inf) if kind == "hold" else np.where(same, np.inf, P)
        if static:
            cache[kind] = P
        return P

    C = E_frames[0][None, :]                   # (copies, states)
    back = []
    for t in range(1, T):
        enter, leave = inside[t] and not inside[t - 1], inside[t - 1] and not inside[t]
        if not free[t]:
            M = C[:, None, :] - (pen(t, "hold") if hold_sign else pen_fn(t, False))[None]
        elif not (inside[t] or inside[t - 1]):  # t1 = t - 1, t2 = t
            M = C[:, None, :] - pen(t, "flip")[None]
        else:
            if enter:                          # split by side at t1
                C = np.where(side[t - 1][None, :] == np.array([[1], [-1]]), C, -np.inf)
            M = C[:, None, :] - pen(t, "free")[None]
        a = M.argmax(-1)                       # (copies, n_t)
        val = np.take_along_axis(M, a[..., None], -1)[..., 0]
        if leave:                              # side at t2 opposite to that at t1
            c, k = (side[t] > 0).astype(int), np.arange(a.shape[1])
            back.append((c[None, :], a[c, k][None, :]))
            val = val[c, k][None, :]
        else:
            back.append((np.broadcast_to(np.arange(a.shape[0])[:, None] * (not enter), a.shape), a))
        C = E_frames[t][None, :] + val
    c, k = np.unravel_index(np.argmax(C), C.shape)
    if not np.isfinite(C[c, k]):
        raise ValueError("no ridge satisfies the zero-crossing constraints (an interval "
                         "too short for max_dk, or hold_sign with a sign change elsewhere)")
    path = np.empty(T, int)
    path[-1] = k
    for t in range(T - 1, 0, -1):
        bc, bi = back[t - 1]
        c, k = bc[c, k], bi[c, k]
        path[t - 1] = k
    return path


def ridge_dp(E, shift, lam_k, lam_b, max_dk, max_db, r_k=8, r_b=4, flips=None,
             side=None, hold_sign=True, lam_b_cross=0.0):
    """Coarse-to-fine DP maximizing
         sum_t E[t, k_t, b_t] - lam_k sum (dk - shift)^2 - lam_b sum db^2,
    shift = mean of shift[b_{t-1}], shift[b_t] (zeros -> paper's cost (14)).
    |dk| <= max_dk, |db| <= max_db. Returns (k_path, b_path).
    flips: optional frame pairs (t1, t2), t1 < t2, sorted, intervals sharing at
    most an end frame, each bracketing one sign change: side[k_t1] = -side[k_t2]
    (side: +1 / -1 per frequency bin); for the moves between t1 and t2 lam_b
    is replaced by lam_b_cross (0: chirp rate unconstrained) and max_db is not
    applied. hold_sign: side[k_t] constant outside the intervals."""
    T, K, B = E.shape
    r_b = min(r_b, B)
    Kc, Bc = int(np.ceil(K / r_k)), int(np.ceil(B / r_b))
    Ep = np.full((T, Kc * r_k, Bc * r_b), -np.inf)
    Ep[:, :K, :B] = E
    blocks = Ep.reshape(T, Kc, r_k, Bc, r_b).transpose(0, 1, 3, 2, 4).reshape(T, Kc, Bc, -1)
    Ec = blocks.max(-1).reshape(T, -1)
    arg = blocks.argmax(-1)
    CK, CB = np.meshgrid(np.arange(Kc) * r_k + (r_k - 1) / 2,
                         np.arange(Bc) * r_b + (r_b - 1) / 2, indexing="ij")
    CK, CB = CK.ravel(), CB.ravel()
    sh_c = np.interp(CB, np.arange(B), shift)
    dk = CK[:, None] - CK[None, :]
    db = CB[:, None] - CB[None, :]
    Pk = lam_k * (dk - 0.5 * (sh_c[:, None] + sh_c[None, :])) ** 2
    Pc = Pk + lam_b * db ** 2
    Pc[(np.abs(dk) > max_dk + r_k) | (np.abs(db) > max_db + r_b)] = np.inf
    if flips is not None:
        Pc_free = Pk + lam_b_cross * db ** 2
        Pc_free[np.abs(dk) > max_dk + r_k] = np.inf
        # side of a coarse cell = side of its best fine cell (the one refined below);
        # constant in time unless a cell straddles 0 Hz
        k_rep = np.arange(Kc)[None, :, None] * r_k + arg // r_b
        side_c = side[np.clip(k_rep, 0, K - 1)].reshape(T, -1)
        pc = _viterbi_flips([Ec[t] for t in range(T)], lambda t, f: Pc_free if f else Pc,
                            side_c, flips, hold_sign, static=bool(np.all(side_c == side_c[0])))
    else:
        pc = _viterbi([Ec[t] for t in range(T)], lambda t: Pc)
    ic, jc = np.divmod(pc, Bc)
    sub = arg[np.arange(T), ic, jc]
    k_cent = np.clip(ic * r_k + sub // r_b, 0, K - 1)
    b_cent = np.clip(jc * r_b + sub % r_b, 0, B - 1)

    LK, LB = np.meshgrid(np.arange(-r_k, r_k + 1), np.arange(-r_b, r_b + 1), indexing="ij")
    Ks = k_cent[:, None] + LK.ravel()[None, :]
    Bs = b_cent[:, None] + LB.ravel()[None, :]
    valid = (Ks >= 0) & (Ks < K) & (Bs >= 0) & (Bs < B)
    Kcl, Bcl = np.clip(Ks, 0, K - 1), np.clip(Bs, 0, B - 1)
    Ef = np.where(valid, E[np.arange(T)[:, None], Kcl, Bcl], -np.inf)
    shf = shift[Bcl]

    def pen(t, free=False):
        dk = Ks[t][:, None] - Ks[t - 1][None, :]
        P = lam_k * (dk - 0.5 * (shf[t][:, None] + shf[t - 1][None, :])) ** 2
        db = Bs[t][:, None] - Bs[t - 1][None, :]
        P += (lam_b_cross if free else lam_b) * db ** 2
        bad = np.abs(dk) > max_dk
        if not free:
            bad |= np.abs(db) > max_db
        P[bad] = np.inf
        return P

    if flips is not None:
        pf = _viterbi_flips([Ef[t] for t in range(T)], pen, side[Kcl], flips, hold_sign)
    else:
        pf = _viterbi([Ef[t] for t in range(T)], pen)
    return Ks[np.arange(T), pf], Bs[np.arange(T), pf]


def ridge_greedy(F, lam_k, lam_b, max_dk, max_db, n_init=10, rng=None):
    """Paper Algorithm 1: from a start frame q take the global maximum, then
    move forward / backward choosing, inside the allowed window around the
    previous point, the cell maximizing F - lam_k dk^2 - lam_b db^2. Several
    start frames; keep the path with the largest cost (14)."""
    rng = np.random.default_rng(0) if rng is None else rng
    T, K, B = F.shape
    dks = np.arange(-max_dk, max_dk + 1)
    dbs = np.arange(-max_db, max_db + 1)
    DK, DB = np.meshgrid(dks, dbs, indexing="ij")
    pen = lam_k * DK ** 2 + lam_b * DB ** 2
    starts = [int(np.argmax(F.reshape(T, -1).max(1)))]
    starts += list(rng.choice(T, size=min(n_init, T) - 1, replace=False))
    best, best_cost = None, -np.inf
    for q in starts:
        kp, bp = np.empty(T, int), np.empty(T, int)
        kp[q], bp[q] = np.unravel_index(np.argmax(F[q]), (K, B))
        for order in (range(q + 1, T), range(q - 1, -1, -1)):
            for n in order:
                prev = n - 1 if n > q else n + 1
                ks = kp[prev] + DK
                bs = bp[prev] + DB
                ok = (ks >= 0) & (ks < K) & (bs >= 0) & (bs < B)
                val = np.where(ok, F[n, np.clip(ks, 0, K - 1), np.clip(bs, 0, B - 1)] - pen, -np.inf)
                i = np.unravel_index(np.argmax(val), val.shape)
                kp[n], bp[n] = ks[i], bs[i]
        cost = (F[np.arange(T), kp, bp].sum() - lam_k * np.sum(np.diff(kp) ** 2)
                - lam_b * np.sum(np.diff(bp) ** 2))
        if cost > best_cost:
            best, best_cost = (kp.copy(), bp.copy()), cost
    return best


def _energy_log(R, blur):
    R = R.astype(np.float64)          # float32 volumes: avoid underflow of the floors
    Rb = gaussian_filter(R, (0, blur[0], blur[1]), mode="nearest") if any(blur) else R
    E = np.log(Rb + 1e-6 * Rb.max(axis=(1, 2), keepdims=True) + 1e-300)
    return E - E.max(axis=(1, 2), keepdims=True)   # empty frames -> all zeros


def _energy_linear(R, blur):
    R = R.astype(np.float64)
    Rb = gaussian_filter(R, (0, blur[0], blur[1]), mode="nearest") if any(blur) else R
    return Rb / np.maximum(Rb.max(axis=(1, 2), keepdims=True), 1e-300)


def _flip_frames(zero_crossings, t):
    """Sample intervals [idx1, idx2] -> sorted frame pairs (t1, t2): nearest
    frame centres t, clipped so that t1 < t2."""
    iv = np.atleast_2d(np.asarray(zero_crossings, dtype=float))
    if iv.size == 0:
        return []
    if iv.ndim != 2 or iv.shape[1] != 2:
        raise ValueError("zero_crossings must be a pair (idx1, idx2) or a list of pairs")
    flips = []
    for a, b in sorted(np.sort(iv, axis=1).tolist()):
        t1 = min(int(np.argmin(np.abs(t - a))), t.size - 2)
        t2 = max(int(np.argmin(np.abs(t - b))), t1 + 1)
        if flips and t1 < flips[-1][1]:
            raise ValueError(f"zero_crossings intervals overlap (frames {flips[-1]} and "
                             f"{(t1, t2)}); give at most one sign change per interval")
        flips.append((t1, t2))
    return flips


def extract_ridges(tfr, n_ridges=1, ridge="dp", lam_k=0.05, lam_b=0.05,
                   max_dk=None, max_db=4, prediction=True, r_k=8, r_b=4,
                   remove=(6, 6), blur=(1.0, 1.0), n_init=10, rng=None,
                   energy="log", zero_crossings=None, hold_sign=True, lam_b_cross=0.0):
    """Peel n_ridges ridges from tfr['R'] (paper Algorithm 1, step 12).
    ridge="dp": coarse-to-fine DP; ridge="greedy": Algorithm 1.
    energy (DP only): "log" (log of blurred energy, frame max = 0) or
    "linear" (blurred energy, frame max = 1, as the greedy detector and
    the paper's cost (14) use). Greedy always uses "linear".
    zero_crossings (DP only, two-sided grid): sample intervals [idx1, idx2],
    each holding one sign change of the IF; every ridge is forced onto opposite
    sides of 0 Hz at idx1 and idx2 (nearest frames), with the chirp rate
    unconstrained in between (lam_b_cross > 0: penalized like lam_b, no
    max_db limit). hold_sign: the intervals hold ALL sign changes, i.e. the
    ridge keeps its side elsewhere. Returns lists of k and b paths."""
    R = tfr["R"].copy()
    T, K, B = R.shape
    hop = tfr["hop"]
    shift = (tfr["beta_out"] * hop / tfr["d_om"]) if (prediction and ridge == "dp") else np.zeros(B)
    if max_dk is None:
        max_dk = int(np.ceil(np.abs(tfr["beta_out"]).max() * hop / tfr["d_om"])) + 6
    max_db = min(max_db, B - 1)
    flips, side = None, None
    if zero_crossings is not None:
        if ridge != "dp":
            raise ValueError("zero_crossings needs ridge='dp'")
        side = np.where(tfr["omega_out"] >= 0, 1, -1)
        if np.all(side > 0):
            raise ValueError("zero_crossings needs a two-sided frequency grid (onesided=False)")
        flips = _flip_frames(zero_crossings, np.asarray(tfr["t"]))
    paths = []
    for _ in range(n_ridges):
        bl = blur if B > 1 else (blur[0], 0)
        if ridge == "dp":
            E = _energy_log(R, bl) if energy == "log" else _energy_linear(R, bl)
            kp, bp = ridge_dp(E, shift, lam_k, lam_b, max_dk, max_db, r_k, r_b,
                              flips, side, hold_sign, lam_b_cross)
        elif ridge == "greedy":
            # Algorithm 1 on linear energy; each frame normalized to max 1 so
            # that lam_k, lam_b have the same meaning in every frame
            F = _energy_linear(R, bl)
            kp, bp = ridge_greedy(F, lam_k, lam_b, max_dk, max_db, n_init, rng)
        else:
            raise ValueError(ridge)
        paths.append((kp, bp))
        for t in range(T):
            R[t, max(0, kp[t] - remove[0]):kp[t] + remove[0] + 1,
              max(0, bp[t] - remove[1]):bp[t] + remove[1] + 1] = 0.0
    return paths


def readout(tfr, kp, bp, dk=2, db=2):
    """Energy-weighted mean of the reassignment operators near the ridge
    (continuous-valued IF / chirp rate in rad/sample, rad/sample^2)."""
    R, Ro, Rb = tfr["R"], tfr["Rw_omega"], tfr["Rw_beta"]
    T, K, B = R.shape
    w = np.empty(T)
    b = np.empty(T)
    for t in range(T):
        ks = np.arange(kp[t] - dk, kp[t] + dk + 1)
        ks = ks % K if np.isclose(K * tfr["d_om"], TWO_PI) else ks[(ks >= 0) & (ks < K)]
        bs = np.arange(max(0, bp[t] - db), min(B, bp[t] + db + 1))
        e = R[t][np.ix_(ks, bs)].sum()
        if e > 0:
            w[t] = Ro[t][np.ix_(ks, bs)].sum() / e
            b[t] = Rb[t][np.ix_(ks, bs)].sum() / e
        else:
            w[t], b[t] = tfr["omega_out"][kp[t]], tfr["beta_out"][bp[t]]
    return w, b


# =========================================================================
# End-to-end
# =========================================================================
def estimate_ifs(x, n_components=1, sigma=10.0, hop=4, nfft=128,
                 n_beta_in=16, beta_max_hz_s=None, n_omega_out=256,
                 n_beta_out=64, mode="joint", gamma=1e-2, onesided=None,
                 ridge="dp", lam_k=0.05, lam_b=0.05, max_dk=None, max_db=4,
                 prediction=True, r_k=8, r_b=4, remove=(6, 6), blur=(1.0, 1.0),
                 readout_cells=(2, 2), n_init=10, energy="log", return_tfr=False,
                 reassign=None, zero_crossings=None, hold_sign=True, lam_b_cross=0.0,
                 *, fs=1.0):
    """IF and chirp rate of n_components ridges.
    fs: sampling rate (keyword, default 1). It only sets the units of the
    output (t in 1 / fs, IF in fs, chirp rate in fs^2: samples, cycles/sample,
    cycles/sample^2 by default) and of beta_max_hz_s (in fs^2; default
    pi / (4 sigma) rad/sample^2, i.e. 1 / (8 sigma) cycles/sample^2).
    onesided: restrict to positive frequencies (default: True for real x
    without zero_crossings).
    reassign: function computing the transform (default chirplet_reassign), e.g.
    functools.partial(tf_fast.chirplet_reassign_fast, backend="processes").
    zero_crossings: optional sign-change hints for a real x on the two-sided
    grid, a pair (idx1, idx2) or a list of pairs of sample indices of x, each
    interval holding exactly one sign change of the IF. The ridge then has
    opposite signs of frequency at idx1 and idx2 (nearest frames) and its chirp
    rate is unconstrained in between (no lam_b penalty, no max_db limit).
    Intervals may share an end but not overlap. ridge="dp" only.
    hold_sign: with zero_crossings, the intervals list ALL sign changes: the
    ridge keeps the sign of its frequency outside them (default; an empty
    list then means no sign change at all). False: only the flip across each
    interval is enforced.
    lam_b_cross: chirp-rate penalty (per bin^2) inside the intervals, replacing
    lam_b there; 0 (default) leaves the chirp rate free. max_db never applies there.
    Returns t, IF (n_components, T), CR (n_components, T) [, tfr]."""
    if onesided is None:
        onesided = np.isrealobj(x) and zero_crossings is None
    beta_max = None if beta_max_hz_s is None else beta_max_hz_s * TWO_PI / fs ** 2
    tfr = (reassign or chirplet_reassign)(x, sigma=sigma, hop=hop, nfft=nfft, n_beta_in=n_beta_in,
                            beta_max=beta_max, n_omega_out=n_omega_out,
                            n_beta_out=n_beta_out, mode=mode, gamma=gamma,
                            omega_range=(0.0, np.pi) if onesided else None)
    paths = extract_ridges(tfr, n_components, ridge, lam_k, lam_b, max_dk, max_db,
                           prediction, r_k, r_b, remove, blur, n_init, energy=energy,
                           zero_crossings=zero_crossings, hold_sign=hold_sign,
                           lam_b_cross=lam_b_cross)
    IF, CR = [], []
    for kp, bp in paths:
        w, b = readout(tfr, kp, bp, *readout_cells)
        IF.append(w * fs / TWO_PI)
        CR.append(b * fs ** 2 / TWO_PI)
    out = (tfr["t"] / fs, np.array(IF), np.array(CR))
    if return_tfr:
        tfr["paths"] = paths
        tfr["ridge_k"], tfr["ridge_b"] = paths[0]
        return out + (tfr,)
    return out


def estimate_if_chirplet(x, **kw):
    """Single-component convenience wrapper of estimate_ifs (fs: keyword, default 1)."""
    ret = estimate_ifs(x, n_components=1, **kw)
    t, IF, CR = ret[:3]
    return (t, IF[0], CR[0]) + tuple(ret[3:])


# =========================================================================
# Closed-form chirplet transform of a unimodular signal (paper eq. 5)
# =========================================================================
def chirplet_closed_form(inst_freq, omega, beta, sigma, phi0=0.0, freq_units="rad",
                         n_alias=1, dtype=np.complex128, t_index=None):
    """Evaluate eq. 5 of Zhu et al. (local quadratic-phase approximation of the
    chirplet transform) for x[n] = exp(j phi[n]), phi[n] = phi0 + 2 pi int IF.

    C(t, w, b) = x(t) sqrt(sigma) sqrt(2 pi / (1 + j sigma^2 (b - phi''(t))))
                 * exp(-sigma^2 (w - phi'(t))^2 / (2 (1 + j sigma^2 (b - phi''(t)))))

    Same conventions as chirplet_reassign (window exp(-tau^2/2 sigma^2)/sqrt(sigma),
    sum_tau x[t+tau] g(tau) exp(-j b tau^2/2) exp(-j w tau), phase referenced to
    the frame centre t), so the two can be compared directly.

    Aliasing (samples at 1 Hz): inst_freq may take any value and is treated as
    UNWRAPPED (e.g. a chirp running past 0.5 cycles/sample keeps increasing).
      - phi is integrated from the unwrapped IF (this is the true signal;
        adding an integer to the IF leaves x[n] unchanged),
      - phi'' is the derivative of the unwrapped IF, so the chirp rate has no
        spikes where the IF crosses +/- 0.5,
      - w - phi' is wrapped into [-pi, pi) (equivalent to wrapping the IF into
        [-0.5, 0.5)), and the 2 pi-periodicity of the discrete-time transform
        is accounted for by summing eq. 5 at w - phi' + 2 pi k,
        k = -n_alias..n_alias (Poisson summation; 1 is enough unless sigma < ~1).
      omega may also lie outside [-pi, pi).

    Parameters
    ----------
    inst_freq : (N1,) IF in cycles/sample (unwrapped, any real values)
    omega     : (N2,) frequency axis; rad/sample (freq_units="rad", like
                omega_in) or cycles/sample (freq_units="cycles")
    beta      : (N3,) chirp-rate axis; rad/sample^2 ("rad") or cycles/sample^2
    sigma     : Gaussian window std in samples
    phi0      : initial phase (rad)
    t_index   : optional sample indices at which to evaluate (e.g. frame
                centres); phi and phi'' are still computed from the full IF
    Returns
    -------
    C : (N1, N2, N3) complex array (N1 = len(t_index) if given)
    Notes: phi'' = d(2 pi IF)/dn by central differences (np.gradient); phi uses
    trapezoidal integration (phi[n+1] - phi[n] = pi (IF[n] + IF[n+1])). The
    approximation ignores higher-order phase terms (IF curvature within the
    window) and the finite signal length (edges).
    """
    f = np.asarray(inst_freq, dtype=float)
    if f.ndim != 1:
        raise ValueError("inst_freq must be 1-D")
    w = np.asarray(omega, dtype=float)
    b = np.asarray(beta, dtype=float)
    if freq_units == "cycles":
        w, b = TWO_PI * w, TWO_PI * b
    elif freq_units != "rad":
        raise ValueError("freq_units must be 'rad' or 'cycles'")
    ddphi = np.gradient(TWO_PI * f)                           # phi'' from the UNWRAPPED IF
    phi = phi0 + np.concatenate(([0.0], np.cumsum(np.pi * (f[1:] + f[:-1]))))
    if t_index is not None:
        t_index = np.asarray(t_index)
        f, ddphi, phi = f[t_index], ddphi[t_index], phi[t_index]
    s2 = float(sigma) ** 2
    D = 1.0 + 1j * s2 * (b[None, None, :] - ddphi[:, None, None])      # (N1, 1, N3)
    amp = np.sqrt(sigma) * np.sqrt(TWO_PI / D)                           # principal branch
    dw = _wrap(w[None, :, None] - TWO_PI * f[:, None, None])             # (N1, N2, 1) in [-pi, pi)
    expo = np.zeros(np.broadcast_shapes(dw.shape, D.shape), dtype=np.complex128)
    for k in range(-n_alias, n_alias + 1):
        expo += np.exp(-s2 * (dw + TWO_PI * k) ** 2 / (2.0 * D))
    return (np.exp(1j * phi)[:, None, None] * amp * expo).astype(dtype, copy=False)


def stft_closed_form(inst_freq, omega, sigma, phi0=0.0, freq_units="rad", n_alias=1,
                     model="first_order", dtype=np.complex128, t_index=None):
    """Closed-form Gaussian-window STFT of x[n] = exp(j phi[n]), phi = phi0 + 2 pi int IF.

    The model name is the order of the local Taylor expansion of the phase:
    model="first_order": phase linear within the window (pure tone), Oberlin et
        al. eq. 6,
        V(t, w) = x(t) g_hat(w - phi'(t)),   g_hat(nu) = sqrt(2 pi sigma) exp(-sigma^2 nu^2 / 2)
        for the window g(tau) = exp(-tau^2 / 2 sigma^2) / sqrt(sigma).
    model="second_order": phase quadratic within the window (local chirp), Zhu et
        al. eq. 5 at beta = 0, i.e. chirplet_closed_form at beta = 0 (includes the
        spreading caused by the chirp rate).
    Same conventions, units, unwrapped-IF handling, aliasing (n_alias) and
    t_index as chirplet_closed_form. Returns (N1, N2) complex."""
    if model == "second_order":
        return chirplet_closed_form(inst_freq, omega, [0.0], sigma, phi0, freq_units, n_alias,
                                    dtype, t_index)[:, :, 0]
    if model != "first_order":
        raise ValueError("model must be 'first_order' or 'second_order'")
    f = np.asarray(inst_freq, dtype=float)
    w = np.asarray(omega, dtype=float) * (TWO_PI if freq_units == "cycles" else 1.0)
    phi = phi0 + np.concatenate(([0.0], np.cumsum(np.pi * (f[1:] + f[:-1]))))
    if t_index is not None:
        t_index = np.asarray(t_index)
        f, phi = f[t_index], phi[t_index]
    s2 = float(sigma) ** 2
    dw = _wrap(w[None, :] - TWO_PI * f[:, None])
    g_hat = sum(np.exp(-s2 * (dw + TWO_PI * k) ** 2 / 2.0) for k in range(-n_alias, n_alias + 1))
    return (np.exp(1j * phi)[:, None] * np.sqrt(TWO_PI * sigma) * g_hat).astype(dtype, copy=False)
