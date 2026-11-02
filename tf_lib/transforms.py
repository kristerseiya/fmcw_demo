
import numpy as np
from . import chirplet_ridge as CR

TRANSFORMS = ("stft", "fsst", "chirplet", "fcrm")
TWO_PI = 2 * np.pi

# ---------------------------------------------------------------- transforms of the data
def compute_transforms(x, fs, sigma, hop, nfft, n_beta_in, max_chirp_hz_s,
                       n_omega_out, n_beta_out, mode, reassign=CR.chirplet_reassign,
                       which=TRANSFORMS):
    """Requested representations on a common frame grid, axes in s / Hz / Hz/s.
    which: any of "stft", "fsst", "chirplet", "fcrm". Real x: positive frequencies
    only. Arrays are indexed [time, frequency(, chirp)].
    reassign: transform function (chirplet_reassign or a tf_fast version).

    STFT and FSST come from one call (mode "fsst"), chirplet and FCRM from another;
    a call is skipped if neither of its outputs is requested, and the reassignment
    (the expensive part) is skipped when only the transform is requested."""
    which = set(which)
    unknown = which - set(TRANSFORMS)
    if unknown or not which:
        raise ValueError(f"which must be a non-empty subset of {TRANSFORMS}, got {sorted(which)}")
    real = np.isrealobj(x)
    band = (0.0, np.pi) if real else None
    max_chirp = max_chirp_hz_s or fs ** 2 / (8 * sigma)
    hz, hz_s = fs / TWO_PI, fs ** 2 / TWO_PI
    common = dict(sigma=sigma, hop=hop, nfft=nfft, omega_range=band, n_omega_out=n_omega_out)

    out = dict(fs=fs, sigma=sigma, nfft=nfft, real=real, which=tuple(w for w in TRANSFORMS if w in which))
    keep = None
    if which & {"stft", "fsst"}:
        st = reassign(x, mode="fsst", return_complex="stft" in which,
                      compute_R="fsst" in which, **common)
        keep = st["omega_in"] >= 0 if real else np.ones(nfft, bool)   # input grid is full band
        out.update(t=st["t"] / fs, t_idx=st["t"], f_in=st["omega_in"][keep] * hz,
                   f_out=st["omega_out"] * hz)
        if "stft" in which:
            out["stft"] = st["C"][:, keep, 0]              # complex, (T, F)
        if "fsst" in which:
            out["fsst"] = st["R"][:, :, 0]                 # energy,  (T, F_out)
    if which & {"chirplet", "fcrm"}:
        ct = reassign(x, mode=mode, n_beta_in=n_beta_in, n_beta_out=n_beta_out,
                      beta_max=max_chirp / hz_s, return_complex="chirplet" in which,
                      compute_R="fcrm" in which, **common)
        keep = ct["omega_in"] >= 0 if real else np.ones(nfft, bool)
        out.update(t=ct["t"] / fs, t_idx=ct["t"], f_in=ct["omega_in"][keep] * hz,
                   f_out=ct["omega_out"] * hz, c_in=ct["beta_in"] * hz_s,
                   c_out=ct["beta_out"] * hz_s)
        if "chirplet" in which:
            out["chirplet"] = ct["C"][:, keep, :]          # complex, (T, F, C)
        if "fcrm" in which:
            out["fcrm"] = ct["R"]                          # energy,  (T, F_out, C_out)
    return out


# ---------------------------------------------------------------- models from the IF
def model_transforms(inst_freq, tf, amps=None, phi0=None, stft_model="first_order",
                     which=None):
    """Expected / ideal representations of x = sum_k amps_k exp(j phi_k), phi_k = phi0_k +
    2 pi int IF_k (or the real part, if the data in tf are real), on the grids of tf.

    inst_freq : (N,) or (K, N) IF in Hz, unwrapped, sampled like the signal
    stft      : stft_model="first_order": phase linear in the window (Oberlin et al. eq. 6),
                "second_order": phase quadratic in the window (Zhu et al. eq. 5 at beta = 0)
    chirplet  : eq. 5 of Zhu et al.
    fsst      : ideal FSST, each frame's STFT-model energy in the bin of IF(t)
    fcrm      : ideal FCRM, each frame's chirplet-model energy in the cell of
                (IF(t), chirp rate(t)), chirp rate = d IF / dt
    Components are summed coherently in stft / chirplet (cross terms included) and
    in energy in fsst / fcrm. A real signal cos(phi) = (e^{j phi} + e^{-j phi}) / 2 is
    modelled as two mirror images; on a one-sided grid only the positive one shows.
    which: models to build (default: the transforms present in tf).
    """
    which = set(tf["which"] if which is None else which)
    fs, sigma, nfft, idx = tf["fs"], tf["sigma"], tf["nfft"], tf["t_idx"]
    IF = np.atleast_2d(np.asarray(inst_freq, float))
    K = IF.shape[0]
    amps = np.ones(K) if amps is None else np.asarray(amps, float)
    phi0 = np.zeros(K) if phi0 is None else np.asarray(phi0, float)
    images = [(IF[k], amps[k], phi0[k]) for k in range(K)]
    if tf["real"]:
        images = [(s * f, a / 2, s * p) for f, a, p in images for s in (1, -1)]

    w_in = tf["f_in"] * TWO_PI / fs
    b_in = tf["c_in"] * TWO_PI / fs ** 2 if "c_in" in tf else None
    w_full = TWO_PI * (np.arange(nfft) - nfft // 2) / nfft         # full band, for energies
    T, tr = idx.size, np.arange(idx.size)
    out = {}
    if "stft" in which:
        out["stft"] = np.zeros((T, w_in.size), complex)
    if "chirplet" in which:
        out["chirplet"] = np.zeros((T, w_in.size, b_in.size), complex)
    if "fsst" in which:
        out["fsst"] = np.zeros((T, tf["f_out"].size))
    if "fcrm" in which:
        out["fcrm"] = np.zeros((T, tf["f_out"].size, tf["c_out"].size))
    for f_hz, a, p in images:
        f = f_hz / fs                                               # cycles/sample
        if "stft" in which:
            out["stft"] += a * CR.stft_closed_form(f, w_in, sigma, p, model=stft_model, t_index=idx)
        if "chirplet" in which:
            out["chirplet"] += a * CR.chirplet_closed_form(f, w_in, b_in, sigma, p, t_index=idx)
        if not which & {"fsst", "fcrm"}:
            continue
        f_wrapped = (f_hz[idx] + fs / 2) % fs - fs / 2              # aliasing
        kf = _bin(f_wrapped, tf["f_out"], wrap=not tf["real"])
        ok = kf >= 0
        if "fsst" in which:
            e_stft = a ** 2 * (np.abs(CR.stft_closed_form(f, w_full, sigma, model=stft_model,
                                                          t_index=idx)) ** 2).sum(1)
            np.add.at(out["fsst"], (tr[ok], kf[ok]), e_stft[ok])
        if "fcrm" in which:
            e_chirp = a ** 2 * (np.abs(CR.chirplet_closed_form(f, w_full, b_in, sigma,
                                                               t_index=idx)) ** 2).sum((1, 2))
            chirp = np.gradient(f_hz, 1 / fs)[idx]                  # from the unwrapped IF
            kc = _bin(chirp, tf["c_out"])
            okc = ok & (kc >= 0)
            np.add.at(out["fcrm"], (tr[okc], kf[okc], kc[okc]), e_chirp[okc])
    return out

def _bin(values, grid, wrap=False):
    """Nearest-bin index on a uniform grid; -1 where outside (or wrapped if wrap)."""
    step = grid[1] - grid[0]
    k = np.rint((values - grid[0]) / step).astype(int)
    if wrap:
        return k % grid.size
    return np.where((k >= 0) & (k < grid.size), k, -1)
