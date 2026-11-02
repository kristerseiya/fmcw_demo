#!/usr/bin/env python3
"""
Analyze ANY signal: FSST + ridge detection, FCRM + 3-D ridge detection and the
Hilbert / phase-difference method, overlaid on the STFT of the data.

Put your data in load_data() below (or pass --npy file.npy --fs 1000).
x may be real or complex:
  real    FSST / FCRM on positive frequencies (FCRM two-sided: --fcrm-two-sided),
          Hilbert analytic signal + phase difference
  complex FSST / FCRM over the full band, phase difference of x itself
FSST and FCRM extract n_ridges ridges by peeling (strongest first; after each
ridge a band of +/- remove_hz (and +/- remove_chirp for FCRM) around it is
zeroed). The phase-difference method gives one estimate (it assumes a single
dominant component).

  python analyze_data.py                          # interactive window
  python analyze_data.py --npy sig.npy --fs 8000 --n-ridges 3
  python analyze_data.py --t-start 2 --t-stop 6 --methods fsst fcrm
  python analyze_data.py --save analysis.png      # no window

Penalties lam (FSST), lam_f / lam_c (FCRM) are per (output-grid bin)^2 of
frame-to-frame change; the frequency grid has n_omega_out bins over the band,
the chirp-rate grid n_beta_out bins over [-max_chirp, max_chirp].
Requires if_lib.py and chirplet_ridge.py in the same folder.
"""
import argparse
import time
import numpy as np
import matplotlib
import scipy
import os
import h5py

this_dir = os.path.dirname(os.path.realpath(__file__))

import tf_lib.if_lib as L
import tf_lib.chirplet_ridge as CR


# =========================================================================
# EDIT ME: load your data. Return (x, fs): 1-D numpy array (real or complex)
# and the sample rate in Hz.
# =========================================================================
def load_data():
    # Example:
    #   x = np.load("my_signal.npy")
    #   fs = 1000.0
    #   return x, fs

    # data = scipy.io.loadmat(os.path.join(this_dir, "..", "data/LNOI_EDBR_manuscript_data_modified/Figure 4/LiDAR/sig_mzi.mat"))
    # signal = data["Sig"]
    # print(signal.shape)
    # signal = signal.reshape((-1,))
    # start = 5464
    # L = 6250*2
    # # i = 8650 + L//2
    # i = 100
    # i = 300
    # x = signal[start+i*L:start+(i+1)*L]
    # fs = 1000

    # f = h5py.File('data/2026_08_21_LiDAR_Measurements/Laser_2/2026_08_21_T17_49_27_Dataset.h5', 'r')
    f = h5py.File('data/2026_08_21_LiDAR_Measurements/Laser_2/2026_08_21_T17_47_36_Dataset.h5', 'r')
    x = np.array(f["/channels/channel_1"])
    start = 2750+1500+100000
    duration = 6250
    start = duration * 200
    x = x[start:start+duration]
    fs = 1000

    # Fallback test signal (paper eq. 3): two real components whose IFs cross.
    # fs = 512.0
    # t = np.arange(512) / fs
    # x = (np.exp(-0.03 * t) * np.sin(2 * np.pi * (-90 * t ** 2 + 220 * t))
    #      + (1 + 0.1 * np.cos(20 * np.pi * t)) * np.cos(2 * np.pi * (10 * np.sin(2 * np.pi * t) + 80 * t)))
    # x = x + 0.3 * np.random.default_rng(0).standard_normal(t.size)
    return x, fs


# =========================================================================
# Defaults (all overridable on the command line / in the window)
# =========================================================================
DEFAULTS = dict(
    methods=["fsst", "fcrm", "pd"],
    n_ridges=2,
    hop=None,                 # frames hop (samples); None -> ~fs/250 (FCRM uses 2x)
    # FSST
    fsst_sigma=8.0,           # Gaussian std (samples)
    fsst_lam=0.002,           # ridge penalty per freq-bin^2
    fsst_n_omega=512,
    # FCRM
    fcrm_sigma=8.0,
    fcrm_lam_f=0.05,          # penalty per freq-bin^2 (jump minus chirp prediction)
    fcrm_lam_c=0.3,           # penalty per chirp-bin^2
    fcrm_mode="eq9",          # eq9 | joint | eq6
    fcrm_prediction=True,
    fcrm_n_omega=256,
    fcrm_n_beta=48,
    fcrm_two_sided=False,     # real data only: search both +f and -f
    max_chirp=None,           # Hz/s; None -> fs^2 / (8 sigma) (paper's default range)
    max_jump_hz=None,         # per frame; None -> max_chirp * hop / fs + 3 bins
    remove_hz=None,           # peeling half-width; None -> 2 * fs / (2 pi sigma)
    remove_chirp=None,        # Hz/s; None -> max_chirp / 6
    # phase difference / Hilbert
    pd_lp_type="fir",         # none | fir | ma | butter
    pd_lp_fc=20.0,            # cutoff (Hz)
    pd_lp_len=0,              # taps; 0 = auto from cutoff
    pd_fold="post",           # real data: none | pre | post (fold into [0, fs/2])
    # display
    stft_len=128,
    dyn_range=40.0,
    cmap="Greys",
)

COLORS = dict(
    fsst=["#1A4FD6", "#00A2FF", "#5D3FD3", "#0B2E83", "#3FA7D6"],
    fcrm=["#FE6100", "#C70039", "#B5651D", "#FF8C42", "#8B2500"],
    pd="#DC267F",
)
LEGEND = dict(fontsize=7, loc="upper right", framealpha=0.9, facecolor="white")


# =========================================================================
# Methods
# =========================================================================
def _hops(p, fs):
    h = p["hop"] or max(1, int(round(fs / 250)))
    return h, 2 * h


def run_fsst(x, fs, p, onesided):
    hop, _ = _hops(p, fs)
    sigma = p["fsst_sigma"]
    K = p["fsst_n_omega"]
    df = (fs / 2 if onesided else fs) / K
    max_chirp = p["max_chirp"] or fs ** 2 / (8 * sigma)
    max_jump = p["max_jump_hz"] or max_chirp * hop / fs + 3 * df
    remove = p["remove_hz"] or 2 * fs / (2 * np.pi * sigma)
    t, IF, _, tfr = CR.estimate_ifs(
        x, fs=fs, n_components=p["n_ridges"], sigma=sigma, hop=hop, nfft=L._nfft_for(sigma),
        n_omega_out=K, mode="fsst", onesided=onesided, ridge="dp", lam_k=p["fsst_lam"],
        max_dk=int(np.ceil(max_jump / df)), prediction=False, r_k=8, r_b=1,
        remove=(max(1, int(round(remove / df))), 0), blur=(1.0, 0.0), readout_cells=(2, 0),
        return_tfr=True)
    return dict(t=t, IF=IF, tfr=tfr)


def run_fcrm(x, fs, p, onesided):
    _, hop = _hops(p, fs)
    sigma = p["fcrm_sigma"]
    K, Bn = p["fcrm_n_omega"], p["fcrm_n_beta"]
    df = (fs / 2 if onesided else fs) / K
    max_chirp = p["max_chirp"] or fs ** 2 / (8 * sigma)
    d_beta = 2 * max_chirp / (Bn - 1)
    max_jump = p["max_jump_hz"] or max_chirp * hop / fs + 3 * df
    remove = p["remove_hz"] or 2 * fs / (2 * np.pi * sigma)
    remove_c = p["remove_chirp"] or max_chirp / 6
    t, IF, CRt, tfr = CR.estimate_ifs(
        x, fs=fs, n_components=p["n_ridges"], sigma=sigma, hop=hop, nfft=L._nfft_for(sigma),
        n_beta_in=16, beta_max_hz_s=max_chirp, n_omega_out=K, n_beta_out=Bn,
        mode=p["fcrm_mode"], onesided=onesided, ridge="dp", lam_k=p["fcrm_lam_f"],
        lam_b=p["fcrm_lam_c"], max_dk=int(np.ceil(max_jump / df)), max_db=max(2, Bn // 4),
        prediction=p["fcrm_prediction"],
        remove=(max(1, int(round(remove / df))), max(1, int(round(remove_c / d_beta)))),
        return_tfr=True)
    return dict(t=t, IF=IF, CR=CRt, tfr=tfr)


def run_pd(x, fs, p, is_real):
    filt = None if p["pd_lp_type"] == "none" else dict(
        type=p["pd_lp_type"], fc=p["pd_lp_fc"], len=(p["pd_lp_len"] or None))
    fold = None if (not is_real or p["pd_fold"] == "none") else p["pd_fold"]
    t, f = L.phasediff_estimate(x, None, filt, "freq", analytic=is_real, fold=fold, fs=fs)
    return dict(t=t, IF=f[None, :])


def analyze(x, fs, p):
    is_real = np.isrealobj(x)
    out = {}
    for m in p["methods"]:
        t0 = time.perf_counter()
        if m == "fsst":
            out[m] = run_fsst(x, fs, p, onesided=is_real)
        elif m == "fcrm":
            out[m] = run_fcrm(x, fs, p, onesided=is_real and not p["fcrm_two_sided"])
        elif m == "pd":
            out[m] = run_pd(x, fs, p, is_real)
        out[m]["time"] = time.perf_counter() - t0
    return out


def fcrm_memory_mb(n, fs, p):
    _, hop = _hops(p, fs)
    return (n // hop + 1) * p["fcrm_n_omega"] * p["fcrm_n_beta"] * 4 * 6 / 1e6


# =========================================================================
# Drawing
# =========================================================================
def show_image(ax, t, f, Z, vmin, vmax, cmap="Greys", alpha=0.8):
    dt = (t[1] - t[0]) if len(t) > 1 else 1.0
    dfr = (f[1] - f[0]) if len(f) > 1 else 1.0
    ax.imshow(Z, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=vmin, vmax=vmax, alpha=alpha,
              extent=[t[0] - dt / 2, t[-1] + dt / 2, f[0] - dfr / 2, f[-1] + dfr / 2])


def draw(axs, x, fs, t_off, res, p, vis):
    a_stft, a_fsst, a_fcrm, a_cr = axs
    for a in axs:
        a.clear()
    is_real = np.isrealobj(x)
    flo = 0 if is_real and not (p["fcrm_two_sided"] and "fcrm" in res) else -fs / 2
    dur = x.size / fs
    ts, fr, S = L.spectrogram(x, fs, L=p["stft_len"], hop=max(1, p["stft_len"] // 8),
                              twosided=not is_real or flo < 0)
    show_image(a_stft, ts + t_off, fr, S, -p["dyn_range"], 0, p["cmap"])
    lines = []
    for m, name, z in (("fsst", "FSST", 3), ("fcrm", "FCRM", 4)):
        if m in res and vis.get(m, True):
            for i, f in enumerate(res[m]["IF"]):
                a_stft.plot(res[m]["t"] + t_off, f, color=COLORS[m][i % 5], lw=1.6, zorder=z,
                            ls="-" if m == "fsst" else "--", label=f"{name} ridge {i + 1}")
    if "pd" in res and vis.get("pd", True):
        lab = "Hilbert + phase diff" if is_real else "phase diff"
        a_stft.plot(res["pd"]["t"] + t_off, res["pd"]["IF"][0], color=COLORS["pd"], lw=0.9,
                    zorder=5, label=f"{lab} ({p['pd_lp_type']}, {p['pd_lp_fc']:.3g} Hz)")
    a_stft.set_ylim(flo, fs / 2)
    a_stft.set_xlim(t_off, t_off + dur)
    a_stft.set_xlabel("time (s)"); a_stft.set_ylabel("frequency (Hz)")
    a_stft.set_title(f"STFT (Hann {p['stft_len']}), {'real' if is_real else 'complex'} data, "
                     f"fs = {fs:g} Hz")
    if a_stft.get_legend_handles_labels()[0]:
        a_stft.legend(**LEGEND)

    for a, m, title in ((a_fsst, "fsst", "FSST"), (a_fcrm, "fcrm", "FCRM (summed over chirp rate)")):
        if m not in res:
            a.text(0.5, 0.5, f"{title}: not run", ha="center", va="center", transform=a.transAxes)
            a.set_xticks([]); a.set_yticks([])
            continue
        tfr = res[m]["tfr"]
        Rm = tfr["R"].sum(-1)
        fo = tfr["omega_out"] * fs / (2 * np.pi)
        show_image(a, tfr["t"] / fs + t_off, fo, 10 * np.log10(Rm.T / Rm.max() + 1e-6),
                   -p["dyn_range"], 0, p["cmap"])
        for i, f in enumerate(res[m]["IF"]):
            a.plot(res[m]["t"] + t_off, f, color=COLORS[m][i % 5], lw=1.0)
        a.set_ylim(fo[0], fo[-1])
        a.set_xlim(t_off, t_off + dur)
        sig = p[m + "_sigma"]
        a.set_title(f"{title}, σ = {sig:.1f}  [{res[m]['time']:.2f} s]", fontsize=9)

    if "fcrm" in res:
        for i, c in enumerate(res["fcrm"]["CR"]):
            a_cr.plot(res["fcrm"]["t"] + t_off, c, ".", ms=2, color=COLORS["fcrm"][i % 5],
                      label=f"ridge {i + 1}")
        a_cr.axhline(0, color="gray", lw=0.6); a_cr.grid(alpha=0.25)
        a_cr.set_xlim(t_off, t_off + dur)
        a_cr.set_title("FCRM chirp rate (Hz/s)", fontsize=9)
        a_cr.legend(**LEGEND)
    a_cr.set_xlabel("time (s)")


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", help="load x from this .npy file instead of load_data()")
    ap.add_argument("--fs", type=float, help="sample rate (required with --npy)")
    ap.add_argument("--t-start", type=float, default=0.0, help="crop start (s)")
    ap.add_argument("--t-stop", type=float, default=None, help="crop stop (s)")
    ap.add_argument("--save", help="render once to this image file, no window")
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                            default=v, metavar="BOOL")
        elif isinstance(v, list):
            ap.add_argument(flag, nargs="+", default=v)
        elif v is None:
            ap.add_argument(flag, type=float, default=None)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

    if a.npy:
        if a.fs is None:
            raise SystemExit("--fs is required with --npy")
        x, fs = np.load(a.npy), float(a.fs)
    else:
        x, fs = load_data()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise SystemExit(f"expected a 1-D signal, got shape {x.shape}")
    i0 = int(round(a.t_start * fs))
    i1 = x.size if a.t_stop is None else int(round(a.t_stop * fs))
    x = x[i0:i1]
    t_off = i0 / fs
    p = {k: getattr(a, k) for k in DEFAULTS}
    if p["hop"] is not None:
        p["hop"] = int(p["hop"])
    print(f"{x.size} samples ({x.size / fs:.2f} s) of {'real' if np.isrealobj(x) else 'complex'} "
          f"data at fs = {fs:g} Hz; methods {p['methods']}")
    if "fcrm" in p["methods"] and fcrm_memory_mb(x.size, fs, p) > 1500:
        print(f"warning: FCRM needs ~{fcrm_memory_mb(x.size, fs, p):.0f} MB; "
              "raise --hop, lower --fcrm-n-beta / --fcrm-n-omega, or crop with --t-start/--t-stop")

    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(3, 2, left=0.06, right=0.98, top=0.96, bottom=0.36,
                          hspace=0.45, wspace=0.12, width_ratios=[1.35, 1])
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])]
    vis = {m: True for m in ("fsst", "fcrm", "pd")}
    last = {"res": analyze(x, fs, p)}
    for m, r in last["res"].items():
        print(f"  {m}: {r['time']:.2f} s")
    draw(axs, x, fs, t_off, last["res"], p, vis)
    if a.save:
        fig.savefig(a.save, dpi=100)
        print(f"saved {a.save}")
        return

    specs = [  # key, label, min, max, init, step, log
        ("n_ridges", "number of ridges", 1, 6, p["n_ridges"], 1, False),
        ("fsst_sigma", "FSST σ (samples)", 2, 64, p["fsst_sigma"], 0.5, False),
        ("fsst_lam", "log10 FSST λ", -5, 1, p["fsst_lam"], 0.1, True),
        ("fcrm_sigma", "FCRM σ (samples)", 2, 64, p["fcrm_sigma"], 0.5, False),
        ("fcrm_lam_f", "log10 FCRM λ freq", -4, 1, p["fcrm_lam_f"], 0.1, True),
        ("fcrm_lam_c", "log10 FCRM λ chirp", -4, 1, p["fcrm_lam_c"], 0.1, True),
        ("pd_lp_fc", "log10 phase-diff LP cutoff (Hz)", -1, np.log10(fs / 2), p["pd_lp_fc"], 0.05, True),
        ("stft_len", "display STFT len", 16, 1024, p["stft_len"], 16, False),
        ("dyn_range", "display dB range", 10, 100, p["dyn_range"], 5, False),
    ]
    sliders = {}
    for i, (k, lab, lo, hi, v, st, lg) in enumerate(specs):
        ax = fig.add_axes([0.17, 0.30 - i * 0.032, 0.36, 0.02])
        sliders[k] = (Slider(ax, lab, lo, hi, valinit=np.log10(v) if lg else v, valstep=st), lg)
    run_chk = CheckButtons(fig.add_axes([0.60, 0.17, 0.09, 0.12]), ["fsst", "fcrm", "pd"],
                           [m in p["methods"] for m in ("fsst", "fcrm", "pd")])
    show_chk = CheckButtons(fig.add_axes([0.70, 0.17, 0.09, 0.12]), ["fsst", "fcrm", "pd"],
                            [True, True, True])
    lp_opts, mode_opts = ["none", "fir", "ma", "butter"], ["eq9", "joint", "eq6"]
    rad_lp = RadioButtons(fig.add_axes([0.80, 0.15, 0.08, 0.14]), lp_opts,
                          active=lp_opts.index(p["pd_lp_type"]))
    rad_mode = RadioButtons(fig.add_axes([0.89, 0.17, 0.08, 0.12]), mode_opts,
                            active=mode_opts.index(p["fcrm_mode"]))
    for xx, lab in ((0.60, "run"), (0.70, "show"), (0.80, "phase-diff LP"), (0.89, "FCRM operator")):
        fig.text(xx, 0.30, lab, fontsize=9)
    btn = Button(fig.add_axes([0.80, 0.04, 0.12, 0.07]), "Run")
    status = fig.text(0.60, 0.06, "adjust parameters, then press Run", fontsize=9)

    def run(_=None):
        for k, (s, lg) in sliders.items():
            v = 10 ** s.val if lg else s.val
            p[k] = int(v) if k in ("n_ridges", "stft_len") else v
        p["methods"] = [m for m, on in zip(("fsst", "fcrm", "pd"), run_chk.get_status()) if on]
        p["pd_lp_type"], p["fcrm_mode"] = rad_lp.value_selected, rad_mode.value_selected
        status.set_text("running ...")
        fig.canvas.draw_idle()
        plt.pause(0.01)
        last["res"] = analyze(x, fs, p)
        draw(axs, x, fs, t_off, last["res"], p, vis)
        status.set_text("  ".join(f"{m}: {r['time']:.2f} s" for m, r in last["res"].items()))
        fig.canvas.draw_idle()

    def toggle(_label):
        vis.update(zip(("fsst", "fcrm", "pd"), show_chk.get_status()))
        draw(axs, x, fs, t_off, last["res"], p, vis)
        fig.canvas.draw_idle()

    btn.on_clicked(run)
    show_chk.on_clicked(toggle)
    plt.show()


if __name__ == "__main__":
    main()
