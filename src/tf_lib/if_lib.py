"""
Shared code for the IF-estimation experiments and demos.

  processes   rect-PSD Gaussian process -> instantaneous frequency (IF)
  signals     complex exp(j phi) or real cos(phi), + white Gaussian noise
  estimators  phase difference (+ optional Hilbert) + low-pass filter
              FSST + coarse-to-fine 2-D ridge   (complex signals)
              FCRM + coarse-to-fine 3-D ridge   (real signals)
  scoring     MSE and truncated MSE
  driver      multiprocessing experiment runner (one task per GP instance)

Units of the estimators (lowpass, phasediff_estimate, fsst_estimate,
fcrm_estimate, turning_points*, signed_hilbert_estimate, apply_method): the
sampling rate is an optional keyword fs, default 1. Frequencies (IF, B,
f_scale, cutoffs, band edges) are then in cycles/sample, times (outputs,
env_smooth, min_sep) in samples and chirp rates in cycles/sample^2. Defaults
are relative to fs, so they hold for any fs; explicit values must be in the
units of the fs that is passed (Hz and s for the true sampling rate).
The signal generators and scoring helpers take fs explicitly.

Requires chirplet_ridge.py in the same folder.
"""
import os
import json
import time
import multiprocessing as mp

import numpy as np
from scipy.signal import hilbert, firwin, butter, sosfiltfilt

from . import chirplet_ridge as CR

TWO_PI = 2 * np.pi


# =========================================================================
# Gaussian-process IF and signals
# =========================================================================
def rect_psd_gp(rng, n, fs, B):
    """Zero-mean, unit-variance stationary Gaussian process with PSD flat on
    |f| <= B (rect). Generated at twice the length and cropped so the FFT's
    circular wrap-around does not correlate the two ends."""
    m = 2 * n
    W = np.fft.rfft(rng.standard_normal(m))
    keep = np.fft.rfftfreq(m, 1 / fs) <= B
    z = np.fft.irfft(W * keep, m)[:n]
    n_kept = 2 * keep.sum() - 1 - (keep[-1] if m % 2 == 0 else 0)
    return z / np.sqrt(n_kept / m)


def if_complex(rng, n, fs, B, extent=0.4):
    """IF with min = -extent*fs and max = +extent*fs exactly (affine map of
    one GP realization)."""
    z = rect_psd_gp(rng, n, fs, B)
    a = extent * fs
    return -a + 2 * a * (z - z.min()) / (z.max() - z.min())


def if_real(rng, n, fs, B, f_center=0.0, extent=0.4):
    """Same normalization as the complex case: one GP realization mapped
    affinely onto [f_center - extent*fs, f_center + extent*fs] (default
    [-0.4 fs, 0.4 fs], so the IF crosses zero). Keep |f_center| + extent*fs
    below fs/2 to avoid aliasing."""
    return f_center + if_complex(rng, n, fs, B, extent)


def phase_from_if(f, fs, phi0=0.0):
    """Trapezoidal integration: phi[n+1] - phi[n] = pi (f[n] + f[n+1]) / fs."""
    inc = np.concatenate(([0.0], 0.5 * (f[1:] + f[:-1])))
    return TWO_PI * np.cumsum(inc) / fs + phi0


def unit_noise(rng, n, complex_valued):
    """White Gaussian noise with unit power (circular if complex)."""
    if complex_valued:
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return rng.standard_normal(n)


def add_noise(s, w, snr_db):
    """s + w scaled so that mean|s|^2 / mean|noise|^2 = SNR (w has unit power)."""
    p = np.mean(np.abs(s) ** 2)
    return s + w * np.sqrt(p / 10 ** (snr_db / 10))


def slope_std(B, f_scale):
    """Std of dIF/dt (Hz/s) for IF = f_scale * (unit rect-PSD GP of bandwidth B)."""
    return f_scale * TWO_PI * B / np.sqrt(3)


def curvature_std(B, f_scale):
    """Std of d^2 IF / dt^2 (Hz/s^2)."""
    return f_scale * TWO_PI ** 2 * B ** 2 / np.sqrt(5)


# =========================================================================
# Low-pass filtering (zero phase)
# =========================================================================
def _odd(n):
    n = int(max(1, round(n)))
    return n if n % 2 else n + 1


def lowpass(v, spec, B=None, *, fs=1.0):
    """Zero-phase low-pass of a real or complex sequence sampled at fs
    (keyword, default 1).
    spec: None (no filter) or dict
      type  "fir"    windowed-sinc FIR (Hamming), applied centred (zero phase)
            "ma"     moving average of length len
            "butter" Butterworth (order, default 4), forward-backward
      fc     cutoff in units of fs (cycles/sample at fs = 1), or fc_rel:
             cutoff = fc_rel * B
      len    FIR / MA length in samples (odd). None or 0 = automatic:
             FIR  len = 4 fs / fc  (Hamming transition width ~ 0.8 fc)
             MA   len = 0.443 fs / fc  (boxcar -3 dB point at fc)
             Always capped at a third of the data length. Ignored by butter.
    """
    if spec is None or spec.get("type", "none") == "none":
        return v
    kind = spec["type"]
    fc = spec.get("fc")
    if fc is None and spec.get("fc_rel") is not None:
        fc = spec["fc_rel"] * B
    if kind in ("fir", "ma"):
        if kind == "fir":
            if fc is None or fc >= fs / 2:
                return v
            n = spec.get("len") or 4 * fs / fc
            n = _odd(min(n, v.size // 3))
            h = firwin(n, fc, fs=fs)
        else:
            n = spec.get("len")
            if not n:                       # auto: -3 dB point of the boxcar at fc
                if fc is None:
                    raise ValueError("moving average needs len or a cutoff (fc / fc_rel)")
                n = 0.443 * fs / fc
            n = _odd(min(n, v.size // 3))
            h = np.ones(n) / n
        pad = n // 2
        vp = np.concatenate([v[pad:0:-1], v, v[-2:-pad - 2:-1]]) if pad else v
        if np.iscomplexobj(vp):
            out = np.convolve(vp.real, h, "same") + 1j * np.convolve(vp.imag, h, "same")
        else:
            out = np.convolve(vp, h, "same")
        return out[pad:pad + v.size]
    if kind == "butter":
        if fc is None or fc >= fs / 2:
            return v
        sos = butter(spec.get("order", 4), fc, fs=fs, output="sos")
        if np.iscomplexobj(v):
            return sosfiltfilt(sos, v.real) + 1j * sosfiltfilt(sos, v.imag)
        return sosfiltfilt(sos, v)
    raise ValueError(f"unknown filter type {kind}")


# =========================================================================
# Estimators. Each returns (t, IF estimate[, extras]); fs is an optional
# keyword (default 1): t in samples, IF in cycles/sample, or s and Hz when
# the true fs is passed.
# =========================================================================
def phasediff_estimate(x, B=None, filt=None, domain="freq", analytic=False,
                       fold=None, *, fs=1.0):
    """Phase difference angle(z[n+1] z*[n]) * fs / 2pi, wrapped to (-fs/2, fs/2].
    B: IF bandwidth for filt["fc_rel"] (units of fs). fs: keyword, default 1.
    analytic=True: z = Hilbert analytic signal of the real x.
    domain="freq":   low-pass the IF estimate (classic)
    domain="phasor": low-pass the complex increments z[n+1] z*[n], then take
                     the angle (robust to wrapping / phase slips)
    fold (real signals, whose IF sign is not identifiable): wrap into
      [0, fs/2] via |.|
      "pre"  fold each raw phase difference, then filter (domain="freq");
             with domain="phasor" there is no raw difference to fold, so
             the result is folded after filtering
      "post" filter first, fold the final estimate
      None   no folding (range (-fs/2, fs/2])"""
    z = hilbert(x) if analytic else np.asarray(x)
    inc = z[1:] * np.conj(z[:-1])
    t = (np.arange(inc.size) + 0.5) / fs
    if domain == "phasor":
        f = np.angle(lowpass(inc, filt, B, fs=fs)) * fs / TWO_PI
        if fold in ("pre", "post"):
            f = np.abs(f)
    else:
        f = np.angle(inc) * fs / TWO_PI
        if fold == "pre":
            f = np.abs(f)
        f = lowpass(f, filt, B, fs=fs)
        if fold == "post":
            f = np.abs(f)
    return t, f


def _nfft_for(sigma):
    L = 2 * int(np.ceil(4 * sigma))
    return max(128, 1 << int(np.ceil(np.log2(L))))


def fsst_estimate(x, B, f_scale, sigma=8.0, lam=0.002, hop=4, n_omega_out=512,
                  r_k=8, return_tfr=False, *, fs=1.0):
    """Two-sided Gaussian-window FSST + coarse-to-fine 2-D DP ridge.
    B, f_scale: IF prior (GP bandwidth and IF scale, units of fs; fs: keyword,
    default 1), used to size the frame-to-frame jump limit.
    lam: penalty per (output frequency bin)^2 of frame-to-frame jump; the
    output grid is fs / n_omega_out per bin."""
    df = fs / n_omega_out
    max_dk = int(np.ceil(6 * slope_std(B, f_scale) * hop / fs / df)) + 4
    ret = CR.estimate_ifs(x, n_components=1, sigma=sigma, hop=hop, nfft=_nfft_for(sigma),
                          n_omega_out=n_omega_out, mode="fsst", onesided=False, ridge="dp",
                          lam_k=lam, max_dk=max_dk, prediction=False, r_k=r_k, r_b=1,
                          blur=(1.0, 0.0), readout_cells=(2, 0), return_tfr=return_tfr, fs=fs)
    t, IF = ret[0], ret[1][0]
    return (t, IF, ret[3]) if return_tfr else (t, IF)


def fcrm_estimate(x, B, f_scale, sigma=8.0, lam_f=0.05, lam_c=0.3, hop=8,
                  n_omega_out=256, n_beta_out=48, n_beta_in=16, mode="eq9",
                  prediction=True, onesided=True, r_k=8, r_b=4, return_tfr=False,
                  zero_crossings=None, hold_sign=True, lam_c_cross=0.0, *, fs=1.0):
    """FCRM (chirplet + frequency-chirprate reassignment) + coarse-to-fine 3-D
    DP ridge. B, f_scale: IF prior (GP bandwidth and IF scale, units of fs;
    fs: keyword, default 1). The chirp-rate grid spans +/- 4 std of the IF
    slope expected for bandwidth B (at least +/- 2e-4 fs^2), so lam_c (per
    chirp-rate bin^2) scales with the process.
    lam_f: penalty per frequency bin^2 of the jump residual (jump minus the
    move predicted by the chirp rate when prediction=True). Frequency bins:
    (fs/2)/n_omega_out when onesided, fs/n_omega_out otherwise.
    zero_crossings, hold_sign: IF sign-change hints (sample intervals), see
    chirplet_ridge.estimate_ifs; need onesided=False. lam_c_cross: lam_c inside
    the intervals (0 = chirp rate unconstrained there).
    Returns t, IF[, chirp rate (units of fs^2), tfr]."""
    s_std = slope_std(B, f_scale)
    beta_max = max(2e-4 * fs ** 2, 4 * s_std)       # floor: 200 Hz/s at fs = 1 kHz
    d_beta = 2 * beta_max / (n_beta_out - 1)
    df = (fs / 2 if onesided else fs) / n_omega_out
    max_dk = int(np.ceil(6 * s_std * hop / fs / df)) + 4
    max_db = int(np.ceil(6 * curvature_std(B, f_scale) * hop / fs / d_beta)) + 2
    max_db = min(max_db, n_beta_out - 1)
    ret = CR.estimate_ifs(x, n_components=1, sigma=sigma, hop=hop, nfft=_nfft_for(sigma),
                          n_beta_in=n_beta_in, beta_max_hz_s=beta_max,
                          n_omega_out=n_omega_out, n_beta_out=n_beta_out, mode=mode,
                          onesided=onesided, ridge="dp", lam_k=lam_f, lam_b=lam_c,
                          max_dk=max_dk, max_db=max_db, prediction=prediction,
                          r_k=r_k, r_b=r_b, return_tfr=return_tfr,
                          zero_crossings=zero_crossings, hold_sign=hold_sign,
                          lam_b_cross=lam_c_cross, fs=fs)
    t, IF, CRt = ret[0], ret[1][0], ret[2][0]
    return (t, IF, CRt, ret[3]) if return_tfr else (t, IF)


def apply_method(mth, x, B, f_scale, hop_default=4, *, fs=1.0):
    """Dispatch one method spec (dict) -> (t, IF). fs: keyword, default 1."""
    kind = mth["kind"]
    if kind == "phasediff":
        return phasediff_estimate(x, B, mth.get("filter"), mth.get("domain", "freq"),
                                  analytic=mth.get("analytic", False), fold=mth.get("fold"),
                                  fs=fs)
    if kind == "fsst":
        return fsst_estimate(x, B, f_scale, **mth.get("params", {}), fs=fs)
    if kind == "fcrm":
        t, f = fcrm_estimate(x, B, f_scale, **mth.get("params", {}), fs=fs)
        return t, np.abs(f)       # |IF| (one-sided grid; abs is a safeguard)
    raise ValueError(kind)


# =========================================================================
# Scoring
# =========================================================================
def errors(t_est, f_est, f_true, fs, hop=4, edge=0.3, absolute=False, wrap=False):
    """Errors on a regular grid (every hop samples), ends excluded."""
    n = f_true.size
    t = np.arange(n) / fs
    tg = np.arange(0, n, hop) / fs
    tg = tg[(tg >= edge) & (tg <= t[-1] - edge)]
    truth = np.interp(tg, t, f_true)
    if absolute:
        truth = np.abs(truth)
    e = np.interp(tg, t_est, f_est) - truth
    if wrap:
        e = (e + fs / 2) % fs - fs / 2
    return tg, e


def metrics(e, trunc_frac=0.9):
    """(MSE, truncated MSE over the trunc_frac smallest |errors|)."""
    e2 = np.sort(e ** 2)
    k = max(1, int(np.floor(trunc_frac * e2.size)))
    return e2.mean(), e2[:k].mean()


# =========================================================================
# Display helper
# =========================================================================
def stft(x, fs, L=128, hop=8, twosided=True, nfft=None):
    """Complex STFT with a Hann window of L samples (frame i centred at
    sample i*hop, zero-padded ends). Phase is referenced to the frame
    centre, as in chirplet_ridge. nfft >= L zero-pads each frame.
    Returns t (s), f (Hz), X (F x T complex); twosided=False keeps
    0..fs/2 (for real x)."""
    nfft = L if nfft is None else int(nfft)
    m = np.arange(L) - L // 2
    g = 0.5 * (1 + np.cos(TWO_PI * m / L))
    n0 = np.arange(0, x.size, hop)
    fr = np.pad(np.asarray(x, complex), (L, L))[n0[:, None] + L + m[None, :]] * g
    buf = np.zeros((n0.size, nfft), complex)
    buf[:, m % nfft] = fr                       # tau = 0 at index 0: centre-referenced phase
    X = np.fft.fft(buf, axis=1).T
    f = np.fft.fftfreq(nfft, 1 / fs)
    if twosided:
        X, f = np.fft.fftshift(X, axes=0), np.fft.fftshift(f)
    else:
        X, f = X[:nfft // 2 + 1], np.abs(f[:nfft // 2 + 1])
    return n0 / fs, f, X


def spectrogram(x, fs, L=128, hop=8, twosided=True):
    """|STFT|^2 in dB (Hann window), normalized to max 0 dB.
    Returns t (s), f (Hz), S (F x T)."""
    t, f, X = stft(x, fs, L, hop, twosided)
    S = 10 * np.log10(np.abs(X) ** 2 + 1e-12)
    return t, f, S - S.max()


# =========================================================================
# Experiment driver (one task = one GP instance, all SNRs and methods)
# =========================================================================
def make_instance(kind, cfg, B, rng):
    n = int(round(cfg["duration"] * cfg["fs"]))
    if kind == "complex":
        f = if_complex(rng, n, cfg["fs"], B, cfg["if_extent"])
        s = np.exp(1j * phase_from_if(f, cfg["fs"], rng.uniform(0, TWO_PI)))
    else:
        f = if_real(rng, n, cfg["fs"], B, cfg["f_center"], cfg["if_extent"])
        s = np.cos(phase_from_if(f, cfg["fs"], rng.uniform(0, TWO_PI)))
    w = unit_noise(rng, n, kind == "complex")
    return f, s, w


def f_scale_of(kind, cfg):
    """Prior IF scale (Hz) used to size jump limits / chirp-rate grids; from
    the configuration, not from the realization."""
    return cfg["if_extent"] * cfg["fs"] / 2          # range ~ 4 std (both kinds)


def run_instance(task):
    kind, b, n, cfg, methods = task
    B = cfg["bandwidths"][b]
    rng = np.random.default_rng([cfg["seed"], b, n])
    f_true, s, w = make_instance(kind, cfg, B, rng)
    fsc = f_scale_of(kind, cfg)
    out = np.full((len(cfg["snrs_db"]), len(methods), 2), np.nan)
    for si, snr in enumerate(cfg["snrs_db"]):
        x = add_noise(s, w, snr)
        for j, mth in enumerate(methods):
            te, fe = apply_method(mth, x, B, fsc, fs=cfg["fs"])[:2]
            _, e = errors(te, fe, f_true, cfg["fs"], cfg["hop"], cfg["edge"],
                          absolute=(kind == "real"), wrap=(kind == "complex"))
            out[si, j] = metrics(e, cfg["trunc_frac"])
    return b, n, out


def _save_atomic(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def run_experiment(kind, cfg, methods, out_prefix, workers=1, resume=False,
                   checkpoint_every=20, verbose=True):
    """Result array (B, S, M, N, 2) saved to <out_prefix>.npy (+ .json)."""
    shape = (len(cfg["bandwidths"]), len(cfg["snrs_db"]), len(methods), cfg["n_sims"], 2)
    npy, js = out_prefix + ".npy", out_prefix + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(npy)), exist_ok=True)
    if resume and os.path.exists(npy):
        res = np.load(npy)
        if res.shape != shape:
            raise ValueError(f"existing shape {res.shape} != configured {shape}")
    else:
        res = np.full(shape, np.nan)
    with open(js, "w") as fh:
        json.dump(dict(kind=kind,
                       axes=["bandwidth", "snr_db", "method", "simulation", "metric"],
                       bandwidths=cfg["bandwidths"], snrs_db=cfg["snrs_db"],
                       methods=methods, metrics=["mse", "trunc_mse"], config=cfg,
                       note=("Hz^2; " + ("signed IF, wrapped error" if kind == "complex"
                                         else "error = estimate - |IF| (sign not "
                                              "identifiable); FCRM estimate |.|, "
                                              "Hilbert phase difference folded to "
                                              "[0, fs/2]"))),
                  fh, indent=2)
    tasks = [(kind, b, n, cfg, methods) for b in range(shape[0]) for n in range(shape[3])
             if not np.all(np.isfinite(res[b, :, :, n, :]))]
    if verbose:
        print(f"{len(tasks)} tasks ({shape[0]} bandwidths x {shape[3]} sims; "
              f"{shape[1]} SNRs x {shape[2]} methods each), {workers} worker(s) -> {npy}")
    t0, done = time.perf_counter(), 0

    def handle(r):
        nonlocal done
        b, n, out = r
        res[b, :, :, n, :] = out
        done += 1
        if done % checkpoint_every == 0 or done == len(tasks):
            _save_atomic(npy, res)
            if verbose:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(tasks)} done, {el:.0f} s, "
                      f"~{el / done * (len(tasks) - done):.0f} s left", flush=True)

    if workers <= 1:
        for task in tasks:
            handle(run_instance(task))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(run_instance, tasks, chunksize=1):
                handle(r)
    _save_atomic(npy, res)
    return res


def summarize(res, cfg, methods, metric=1):
    names = [m["name"] for m in methods]
    print(f"\nmedian over simulations of sqrt({['mse', 'trunc_mse'][metric]}) (Hz)")
    for si, snr in enumerate(cfg["snrs_db"]):
        print(f"SNR {snr} dB")
        print(f"{'B (Hz)':>8} " + " ".join(f"{s[:16]:>16}" for s in names))
        for b, B in enumerate(cfg["bandwidths"]):
            v = np.sqrt(np.nanmedian(res[b, si, :, :, metric], axis=1))
            print(f"{B:>8} " + " ".join(f"{x:16.2f}" for x in v))


# =========================================================================
# Signed IF for REAL signals (sign changes = IF zero crossings)
# =========================================================================
def _band_envelope(x, f_lo, f_hi, order, env_smooth, fs):
    """Envelope |hilbert(.)| of x band-passed to [f_lo, f_hi] (Butterworth,
    zero phase), smoothed by a moving average of env_smooth (units of 1 / fs).
    None: f_lo = 0.05 fs, f_hi = 0.45 fs, env_smooth = 15 samples."""
    f_lo = 0.05 * fs if f_lo is None else f_lo
    f_hi = 0.45 * fs if f_hi is None else f_hi
    env_smooth = 15 / fs if env_smooth is None else env_smooth
    sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    n = _odd(env_smooth * fs)
    if n > 1:
        env = np.convolve(np.pad(env, n // 2, mode="reflect"), np.ones(n) / n, "valid")
    return env


def _envelope_dips(env, prominence, thr, min_sep, max_turns, fs):
    """Dips of the (median-normalized) envelope; min_sep in units of 1 / fs
    (None: 30 samples)."""
    from scipy.signal import find_peaks
    min_sep = 30 / fs if min_sep is None else min_sep
    idx, props = find_peaks(-env, prominence=prominence, height=-thr,
                            distance=max(1, int(min_sep * fs)))
    if max_turns and idx.size > max_turns:
        keep = np.argsort(props["prominences"])[::-1][:max_turns]
        idx = np.sort(idx[keep])
    return idx


def turning_points(x, f_lo=None, f_hi=None, order=4, env_smooth=None,
                   prominence=0.5, thr=0.6, min_sep=None, max_turns=None, *, fs=1.0):
    """Detect IF sign changes of a real signal from dips of a high-passed
    envelope. fs: keyword, default 1; frequencies are in units of fs and
    times in units of 1 / fs (cycles/sample and samples at fs = 1).

    The signal is band-pass filtered to [f_lo, f_hi] (Butterworth, zero
    phase; defaults 0.05 fs and 0.45 fs), so while |IF| < f_lo its energy is
    rejected and the envelope |hilbert(filtered)| dips. The envelope is
    smoothed (moving average of env_smooth, default 15 samples) and divided
    by its median. A dip is a local minimum whose prominence (depth relative
    to the surrounding envelope, in units of the median) is >= prominence and
    whose level is < thr; dips closer than min_sep (default 30 samples) are
    merged (scipy.signal.find_peaks on -envelope).
    max_turns: keep only the max_turns most prominent dips (prior knowledge
    of the number of sign changes); None or 0 = keep all.
    Returns t, normalized envelope, turning-point sample indices."""
    env = _band_envelope(x, f_lo, f_hi, order, env_smooth, fs)
    env = env / np.median(env)
    idx = _envelope_dips(env, prominence, thr, min_sep, max_turns, fs)
    return np.arange(x.size) / fs, env, idx


def turning_points_with_locations(x, loc: list = None, f_lo=None, f_hi=None, order=4,
                                  env_smooth=None, prominence=0.5, thr=0.6, min_sep=None,
                                  max_turns=None, *, fs=1.0):
    """turning_points, or, if loc (sample intervals [idx1, idx2]) is given,
    one turning point per interval at the minimum of the (not normalized)
    envelope inside it. Same parameters and units as turning_points."""
    env = _band_envelope(x, f_lo, f_hi, order, env_smooth, fs)
    if loc:
        idx = []
        for loci in loc:
            idx1 = max(loci[0], 0)
            idx2 = min(loci[1], len(x))
            tpidx = np.argmin(env[idx1:idx2]) + idx1
            idx.append(tpidx)
    else:
        env = env / np.median(env)
        idx = _envelope_dips(env, prominence, thr, min_sep, max_turns, fs)
    return np.arange(x.size) / fs, env, idx


def sign_sequence(n, idx):
    """+1 / -1 per sample, alternating at each turning point index."""
    s = np.ones(n)
    for k in idx:
        s[k:] *= -1
    return s


def signed_hilbert_estimate(x, B=None, filt=None, sign_before_filter=True,
                            tp_params=None, *, fs=1.0):
    """Signed IF of a real signal: Hilbert phase differences folded into
    [0, fs/2], multiplied by the turning-point sign sequence, low-passed.
    fs: keyword, default 1 (t in samples, IF in cycles/sample); B and
    tp_params (turning_points_with_locations keywords) in the same units.
    sign_before_filter=True: sign the raw phase differences, then filter
    (signed noise averages out near 0 Hz); False: filter |IF|, then sign.
    The global sign is left arbitrary (+1 before the first turning point).
    Returns t, signed IF, (t_env, env, turning-point indices)."""
    t_env, env, idx = turning_points_with_locations(x, **(tp_params or {}), fs=fs)
    z = hilbert(x)
    # f = np.abs(np.angle(z[1:] * np.conj(z[:-1]))) * fs / TWO_PI
    f = np.angle(z[1:] * np.conj(z[:-1])) * fs / TWO_PI
    t = (np.arange(f.size) + 0.5) / fs
    sg = sign_sequence(x.size, idx)[:-1]
    if sign_before_filter:
        f = lowpass(sg * f, filt, B, fs=fs)
    else:
        f = sg * lowpass(f, filt, B, fs=fs)
    return t, f, (t_env, env, idx)


def best_global_sign(t_est, f_est, f_true, fs, hop=4, edge=0.3):
    """Multiply the estimate by -1 if that lowers the MSE against the signed
    true IF (oracle resolution of the global sign ambiguity)."""
    _, e1 = errors(t_est, f_est, f_true, fs, hop, edge)
    _, e2 = errors(t_est, -f_est, f_true, fs, hop, edge)
    return (1, f_est) if np.mean(e1 ** 2) <= np.mean(e2 ** 2) else (-1, -f_est)


# =========================================================================
# IF with exactly one zero crossing (GP-like segments + cubic bridge)
# =========================================================================
def if_one_crossing(rng, n, fs, B, seg_frac=0.4, offset=0.25, half_range=0.15,
                    direction="random", max_abs=0.45, max_tries=500):
    """IF that changes sign exactly once.

    First seg_frac of the samples: GP1 mapped onto [-offset - half_range,
    -offset + half_range] * fs; last seg_frac: GP2 (independent) mapped onto
    [offset - half_range, offset + half_range] * fs (defaults: [-0.40, -0.10] fs
    and [0.10, 0.40] fs). The middle part is a cubic Hermite bridge matching
    value and slope at both ends (C1 joins, lowest-degree smooth connection).
    A draw is rejected and redrawn unless
      - the IF has exactly one sign change,
      - the bridge is monotone wherever |IF| < (offset - half_range) fs
        (no approach towards 0 Hz that turns back),
      - |IF| <= max_abs * fs everywhere.
    direction: "up" (negative -> positive), "down", or "random".
    Returns IF (Hz) and the number of draws used."""
    n_seg = int(round(seg_frac * n))
    n_mid = n - 2 * n_seg
    if n_mid < 4:
        raise ValueError("seg_frac too large: no room for the bridge")
    gap = (offset - half_range) * fs
    for tries in range(1, max_tries + 1):
        segs = []
        for sgn in (-1, 1):
            z = rect_psd_gp(rng, n_seg, fs, B)
            z = (z - z.min()) / (z.max() - z.min())                 # [0, 1]
            segs.append(sgn * offset * fs + (2 * z - 1) * half_range * fs)
        s1, s2 = segs
        # end values / slopes (Hz per sample), from the last / first two samples
        y0, y1 = s1[-1], s2[0]
        m0, m1 = s1[-1] - s1[-2], s2[1] - s2[0]
        h = n_mid + 1                                               # samples from s1[-1] to s2[0]
        u = np.arange(1, n_mid + 1) / h
        h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
        h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        bridge = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        f = np.concatenate([s1, bridge, s2])
        if np.sum(np.diff(np.sign(f)) != 0) != 1:
            continue
        if np.abs(f).max() > max_abs * fs:
            continue
        near = np.abs(bridge) < gap
        if near.any():
            d = np.diff(bridge)[near[:-1] & near[1:]]
            if d.size and not (np.all(d > 0) or np.all(d < 0)):
                continue
        if direction == "down" or (direction == "random" and rng.random() < 0.5):
            f = -f
        return f, tries
    raise RuntimeError(f"no valid one-crossing IF in {max_tries} draws; "
                       "lower B or widen the bridge (smaller seg_frac)")
