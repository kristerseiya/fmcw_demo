#!/usr/bin/env python3
"""
Interactive demo: signed IF of a REAL signal when the locations of the IF sign
changes (zero crossings) are roughly known.

IF: rect-PSD GP mapped onto [f_center - 0.4 fs, f_center + 0.4 fs] (default
centre 0), so it changes sign; same signal as demo_real_signed.py for the same
seed. Errors are against the SIGNED true IF.

Hint intervals [idx1, idx2] (samples), one per sign change, centre +/- margin
(ends pulled in to the midpoint between neighbouring centres, so the
intervals do not overlap). Centres from
  "true"      the true sign changes, optionally shifted by a random jitter
  "detected"  dips of the band-passed envelope (if_lib.turning_points); a
              near-DC turn-back of the IF also gives a dip, i.e. a wrong hint
FCRM + 3-D ridge, two-sided, without hints: as in demo_real_signed.py.
FCRM + 3-D ridge, two-sided, with hints (chirplet_ridge.estimate_ifs(...,
  zero_crossings=intervals)): the DP ridge must be on opposite sides of 0 Hz
  at idx1 and idx2 of every interval, with the chirp rate unconstrained in
  between ("CR in interval" = free) or penalized by λ chirp; "hold sign" on:
  the intervals hold all sign changes, so the ridge keeps its side elsewhere.
Hilbert, signed: the turning point is the envelope minimum inside each hint
  interval (if_lib.turning_points_with_locations), then as in
  demo_real_signed.py.
Global sign (unidentifiable for a real signal): every estimate is multiplied
by -1 if that lowers the error (oracle).

Panels: STFT (two-sided) + signed IF overlays | FCRM summed over chirp rate |
band-passed envelope with true sign changes and hint intervals | signed error.
The signal is simulated at --fs (Hz); the estimators run at their default
fs = 1, so their parameters are in samples and cycles/sample (hint margin and
jitter, f_lo, envelope smoothing, min separation) and the GP prior is passed
as B / fs. Results are converted back to s and Hz for display and scoring.
Adjust sliders / radio buttons, then press "Run".

  python demo_real_signed_hint.py
  python demo_real_signed_hint.py --B 2 --seed 2 --margin 80 --jitter 20
  python demo_real_signed_hint.py --hint-source detected --save hint.png   # no window
"""
import argparse
import time
import numpy as np
import matplotlib

import tf_lib.if_lib as L

STYLE = dict(
    truth=dict(color="#FFB000", lw=6.0, alpha=0.5, solid_capstyle="round", zorder=2),
    plain=dict(color="#785EF0", lw=1.5, alpha=0.9, ls="--", zorder=3),
    hint=dict(color="#1A4FD6", lw=2.0, alpha=0.9, zorder=4),
    pd=dict(color="#DC267F", lw=1.0, alpha=1.0, zorder=5),
)
SPAN = dict(color="#22A884", alpha=0.18, lw=0, zorder=1)
LEGEND = dict(fontsize=7, loc="upper right", framealpha=0.9, facecolor="white")
NAMES = dict(h="Hilbert (hint)", c="FCRM (no hint)", z="FCRM (hint)")
KEY_STYLE = dict(h="pd", c="plain", z="hint")


def show_image(ax, t, f, Z, vmin, vmax, cmap="Greys", alpha=0.8):
    dt = (t[1] - t[0]) if len(t) > 1 else 1.0
    dfr = (f[1] - f[0]) if len(f) > 1 else 1.0
    ax.imshow(Z, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=vmin, vmax=vmax, alpha=alpha,
              extent=[t[0] - dt / 2, t[-1] + dt / 2, f[0] - dfr / 2, f[-1] + dfr / 2])


def sign_agreement(t_est, f_est, f_true, fs, edge):
    tg, _ = L.errors(t_est, f_est, f_true, fs, edge=edge)
    t = np.arange(f_true.size) / fs
    return np.mean(np.sign(np.interp(tg, t_est, f_est)) == np.sign(np.interp(tg, t, f_true)))


def hint_intervals(centres, margin, n):
    """[c - margin, c + margin] (samples) per centre, ends pulled in to the
    midpoint between neighbouring centres and clipped to [0, n - 1]."""
    c = np.sort(np.asarray(centres, dtype=float))
    if c.size == 0:
        return []
    mid = 0.5 * (c[1:] + c[:-1])
    lo = np.clip(np.maximum(c - margin, np.r_[0.0, mid]), 0, n - 1)
    hi = np.clip(np.minimum(c + margin, np.r_[mid, n - 1.0]), 0, n - 1)
    return [(int(a), int(b)) for a, b in zip(np.rint(lo), np.rint(hi)) if b > a]


def compute(p):
    fs, n = p["fs"], int(round(p["duration"] * p["fs"]))
    rng = np.random.default_rng(p["seed"])
    f = L.if_real(rng, n, fs, p["B"], p["f_center"], p["if_extent"])
    f_scale = p["if_extent"] * fs / 2
    s = np.cos(L.phase_from_if(f, fs, rng.uniform(0, 2 * np.pi)))
    x = L.add_noise(s, L.unit_noise(rng, n, False), p["snr"])
    zc = np.nonzero(np.diff(np.sign(f)) != 0)[0]
    B_n, f_scale_n = p["B"] / fs, f_scale / fs          # estimators at fs = 1: cycles/sample

    tp = dict(f_lo=p["f_lo"], order=p["bp_order"], env_smooth=p["env_smooth"],
              prominence=p["prominence"], thr=p["thr"], min_sep=p["min_sep"])
    te, env, dips = L.turning_points(x, **tp)
    te = te / fs                                        # samples -> s
    if p["source"] == "true":
        jit = np.random.default_rng([p["seed"], 1]).uniform(-1, 1, zc.size)   # signal unchanged
        centres = zc + jit * p["jitter"]
    else:
        centres = dips
    iv = hint_intervals(centres, p["margin"], n)

    filt = None if p["lp_type"] == "none" else dict(type=p["lp_type"], fc_rel=p["lp_fc_rel"])
    t0 = time.perf_counter()
    th, fh, (_, _, idx_h) = L.signed_hilbert_estimate(x, B_n, filt, True, dict(tp, loc=iv))
    t1 = time.perf_counter()
    fk = dict(sigma=p["sigma"], lam_f=10 ** p["log_lam_f"], lam_c=10 ** p["log_lam_c"],
              mode=p["mode"], prediction=p["prediction"], onesided=False, n_omega_out=512)
    tc, fc, _, tfr = L.fcrm_estimate(x, B_n, f_scale_n, return_tfr=True, **fk)
    t2 = time.perf_counter()
    th, fh, tc, fc = th / fs, fh * fs, tc / fs, fc * fs   # samples -> s, cycles/sample -> Hz
    lam_cross = 10 ** p["log_lam_c"] if p["cr_cross"] == "λ chirp" else 0.0
    try:
        tz, fz = L.fcrm_estimate(x, B_n, f_scale_n, zero_crossings=iv, hold_sign=p["hold"],
                                 lam_c_cross=lam_cross, **fk)
        tz, fz, fail = tz / fs, fz * fs, None
    except ValueError as e:                     # infeasible hints (e.g. interval too short)
        tz, fz, fail = tc, np.full_like(fc, np.nan), str(e)
    t3 = time.perf_counter()

    res = dict(t=np.arange(n) / fs, f=f, x=x, tfr=tfr, te=te, env=env, idx_h=idx_h, iv=iv,
               zc=zc, fail=fail, time=dict(h=t1 - t0, c=t2 - t1, z=t3 - t2))
    for k, (tt, fe) in (("h", (th, fh)), ("c", (tc, fc)), ("z", (tz, fz))):
        sg, fe = L.best_global_sign(tt, fe, f, fs, edge=p["edge"])
        tg, e = L.errors(tt, fe, f, fs, edge=p["edge"])          # signed, not wrapped
        res[k] = dict(t=tt, f=fe, sg=sg, tg=tg, e=e, m=np.sqrt(L.metrics(e)),
                      ok=sign_agreement(tt, fe, f, fs, p["edge"]))
    return res


def draw(axs, res, p):
    a0, a1, a2, a3 = axs
    for a in axs:
        a.clear()
    fs, vis = p["fs"], p["show"]
    spans = [(a / fs, b / fs) for a, b in res["iv"]]

    ts, fr, S = L.spectrogram(res["x"], fs, L=p["stft_len"], hop=max(1, p["stft_len"] // 8),
                              twosided=True)
    show_image(a0, ts, fr, S, -p["dyn_range"], 0, p["cmap"])
    tfr = res["tfr"]
    Rs = tfr["R"].sum(-1)
    show_image(a1, tfr["t"] / fs, tfr["omega_out"] * fs / (2 * np.pi),
               10 * np.log10(Rs.T / Rs.max() + 1e-6), -p["dyn_range"], 0, p["cmap"])
    for a in (a0, a1):
        for i, (u, v) in enumerate(spans):
            a.axvspan(u, v, label="hint interval" if i == 0 else None, **SPAN)
        if vis["truth"]:
            a.plot(res["t"], res["f"], label="true IF (signed)", **STYLE["truth"])
        for k in ("c", "z", "h"):
            if vis[k]:
                a.plot(res[k]["t"], res[k]["f"], label=f"{NAMES[k]} (global sign {res[k]['sg']:+d})",
                       **STYLE[KEY_STYLE[k]])
        a.set_ylim(-fs / 2, fs / 2); a.set_xlim(0, p["duration"])
        a.axhline(0, color="k", lw=0.5, alpha=0.5)
        a.set_xlabel("time (s)")
    a0.set_ylabel("frequency (Hz)")
    a0.set_title(f"STFT (Hann {p['stft_len']}), B = {p['B']:.2g} Hz, SNR = {p['snr']:.0f} dB")
    a0.legend(**LEGEND)
    cr = "free" if p["cr_cross"] == "free" else "λ chirp"
    a1.set_title(f"FCRM two-sided ({p['mode']}, σ = {p['sigma']:.1f}); λf = 10^{p['log_lam_f']:.1f}, "
                 f"λc = 10^{p['log_lam_c']:.1f}; hint: hold sign {'on' if p['hold'] else 'off'}, "
                 f"CR in interval {cr}", fontsize=9)
    if res["fail"]:
        a1.text(0.02, 0.95, "hinted ridge: " + res["fail"], transform=a1.transAxes, fontsize=7,
                color="#B00020", va="top", wrap=True)

    a2.plot(res["te"], res["env"], color="#555555", lw=0.8, label="band-passed envelope / median")
    for i, (u, v) in enumerate(spans):
        a2.axvspan(u, v, label="hint interval" if i == 0 else None, **SPAN)
    for i, k in enumerate(res["zc"]):
        a2.axvline(k / fs, color=STYLE["truth"]["color"], lw=3, alpha=0.6,
                   label="true sign change" if i == 0 else None)
    idx = np.asarray(res["idx_h"], dtype=int)
    a2.plot(idx / fs, res["env"][idx], "x", color=STYLE["pd"]["color"], ms=8, mew=2,
            label=f"Hilbert turning points ({idx.size} vs {len(res['zc'])} true)")
    a2.set_ylim(0, max(2.5, 1.1 * np.percentile(res["env"], 99)))
    a2.set_xlim(0, p["duration"])
    src = (f"true sign changes, jitter ±{p['jitter']:.0f} samples" if p["source"] == "true"
           else f"envelope dips (prominence {p['prominence']:.2f}, cap {p['thr']:.2f})")
    a2.set_title(f"hints: {len(res['iv'])} intervals, ±{p['margin']:.0f} samples around {src}",
                 fontsize=9)
    a2.set_xlabel("time (s)"); a2.legend(**LEGEND)

    for k in ("h", "c", "z"):
        r = res[k]
        a3.plot(r["tg"], r["e"], color=STYLE[KEY_STYLE[k]]["color"], lw=1.0,
                ls=STYLE[KEY_STYLE[k]].get("ls", "-"), zorder=STYLE[KEY_STYLE[k]]["zorder"],
                label=f"{NAMES[k]}: RMSE {r['m'][0]:.1f}, trunc {r['m'][1]:.1f} Hz, "
                      f"sign correct {r['ok']:.0%}")
    a3.axhline(0, color="gray", lw=0.6); a3.grid(alpha=0.25)
    m = [res[k]["m"][1] for k in ("h", "c", "z") if np.isfinite(res[k]["m"][1])]
    lim = 5 * max(m + [2.0])
    a3.set_ylim(-min(lim, fs), min(lim, fs))
    tm = res["time"]
    a3.set_title(f"signed error (estimate - IF), {p['edge']} s ends excluded  [Hilbert "
                 f"{tm['h'] * 1e3:.0f} ms, FCRM {tm['c'] * 1e3:.0f} / {tm['z'] * 1e3:.0f} ms]",
                 fontsize=9)
    a3.set_xlabel("time (s)"); a3.set_ylabel("Hz"); a3.legend(**LEGEND)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=float, default=1.0, help="GP bandwidth (Hz)")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=5)
    # hints
    ap.add_argument("--hint-source", default="true", choices=["true", "detected"])
    ap.add_argument("--margin", type=float, default=50.0,
                    help="interval = centre +/- margin (samples)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="random shift of the true centres (samples, hint source 'true')")
    ap.add_argument("--no-hold", action="store_true",
                    help="enforce only the flip across each interval, not a fixed sign elsewhere")
    ap.add_argument("--cr-cross", default="free", choices=["free", "lam-c"],
                    help="chirp rate inside the intervals: free, or penalized by lam_c")
    # FCRM
    ap.add_argument("--sigma", type=float, default=8.0, help="FCRM Gaussian std (samples)")
    ap.add_argument("--lam-f", type=float, default=0.05)
    ap.add_argument("--lam-c", type=float, default=0.3)
    ap.add_argument("--mode", default="eq9", choices=["eq9", "joint", "eq6"])
    ap.add_argument("--no-prediction", action="store_true")
    # Hilbert IF filter
    ap.add_argument("--lp-type", default="fir", choices=["none", "fir", "ma", "butter"])
    ap.add_argument("--lp-fc-rel", type=float, default=4.0, help="cutoff = this * B")
    # envelope (detected hints, display; estimator units: cycles/sample, samples)
    ap.add_argument("--f-lo", type=float, default=0.05,
                    help="band-pass low edge (cycles/sample; 0.05 = 50 Hz at fs = 1 kHz)")
    ap.add_argument("--bp-order", type=int, default=4)
    ap.add_argument("--env-smooth", type=float, default=15.0,
                    help="envelope moving-average length (samples)")
    ap.add_argument("--prominence", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.6, help="dip level cap (x median)")
    ap.add_argument("--min-sep", type=float, default=30.0,
                    help="minimum separation of detected dips (samples)")
    # signal / display
    ap.add_argument("--fs", type=float, default=1000.0,
                    help="simulation sampling rate (Hz); the estimators run at fs = 1")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--f-center", type=float, default=0.0)
    ap.add_argument("--if-extent", type=float, default=0.4)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--stft-len", type=int, default=128)
    ap.add_argument("--cmap", default="Greys")
    ap.add_argument("--dyn-range", type=float, default=40.0)
    ap.add_argument("--save", help="render once to this image file, no window")
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

    p = dict(B=a.B, snr=a.snr, seed=a.seed, source=a.hint_source, margin=a.margin,
             jitter=a.jitter, hold=not a.no_hold,
             cr_cross="free" if a.cr_cross == "free" else "λ chirp",
             sigma=a.sigma, log_lam_f=np.log10(a.lam_f), log_lam_c=np.log10(a.lam_c),
             mode=a.mode, prediction=not a.no_prediction, lp_type=a.lp_type,
             lp_fc_rel=a.lp_fc_rel, f_lo=a.f_lo, bp_order=a.bp_order,
             env_smooth=a.env_smooth, prominence=a.prominence, thr=a.thr,
             min_sep=a.min_sep, fs=a.fs, duration=a.duration, f_center=a.f_center,
             if_extent=a.if_extent, edge=a.edge, stft_len=a.stft_len, cmap=a.cmap,
             dyn_range=a.dyn_range, show=dict(truth=True, h=True, c=True, z=True))

    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.96, bottom=0.44,
                          hspace=0.32, wspace=0.15)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    last = {"res": compute(p)}
    draw(axs, last["res"], p)
    if a.save:
        fig.savefig(a.save, dpi=100)
        print(f"saved {a.save}")
        return

    specs = [  # key, label, min, max, init, step
        ("log_B", "log10 B (Hz)", -1.0, 1.5, np.log10(p["B"]), None),
        ("snr", "SNR (dB)", -10, 40, p["snr"], 1),
        ("seed", "seed", 0, 99, p["seed"], 1),
        ("margin", "hint margin (samples)", 5, 300, p["margin"], 5),
        ("jitter", "hint jitter (samples, true)", 0, 200, p["jitter"], 5),
        ("sigma", "FCRM σ (samples)", 3, 20, p["sigma"], 0.5),
        ("log_lam_f", "log10 λ freq", -4, 1, p["log_lam_f"], 0.1),
        ("log_lam_c", "log10 λ chirp", -4, 1, p["log_lam_c"], 0.1),
        ("lp_fc_rel", "Hilbert LP cutoff / B", 0.5, 12, p["lp_fc_rel"], 0.5),
        ("f_lo", "band-pass f_lo (cyc/sample)", 0.005, 0.2, p["f_lo"], 0.001),
        ("env_smooth", "envelope smooth (samples)", 1, 60, p["env_smooth"], 1),
        ("prominence", "dip prominence", 0.05, 2.0, p["prominence"], 0.05),
        ("thr", "dip level cap", 0.1, 1.5, p["thr"], 0.05),
        ("min_sep", "min separation (samples)", 5, 300, p["min_sep"], 5),
        ("stft_len", "display STFT len", 16, 512, p["stft_len"], 16),
    ]
    sliders = {}
    for i, (k, lab, lo, hi, v, st) in enumerate(specs):
        ax = fig.add_axes([0.14, 0.395 - i * 0.025, 0.40, 0.017])
        sliders[k] = Slider(ax, lab, lo, hi, valinit=v, valstep=st)
    src_opts, cr_opts = ["true", "detected"], ["free", "λ chirp"]
    mode_opts, onoff = ["eq9", "joint", "eq6"], ["on", "off"]
    rad_src = RadioButtons(fig.add_axes([0.60, 0.29, 0.07, 0.09]), src_opts,
                           active=src_opts.index(p["source"]))
    rad_hold = RadioButtons(fig.add_axes([0.68, 0.29, 0.07, 0.09]), onoff,
                            active=0 if p["hold"] else 1)
    rad_cr = RadioButtons(fig.add_axes([0.76, 0.29, 0.07, 0.09]), cr_opts,
                          active=cr_opts.index(p["cr_cross"]))
    rad_mode = RadioButtons(fig.add_axes([0.84, 0.27, 0.06, 0.11]), mode_opts,
                            active=mode_opts.index(p["mode"]))
    rad_pred = RadioButtons(fig.add_axes([0.91, 0.29, 0.07, 0.09]), onoff,
                            active=0 if p["prediction"] else 1)
    for xx, lab in ((0.60, "hint source"), (0.68, "hold sign"), (0.76, "CR in interval"),
                    (0.84, "FCRM operator"), (0.91, "CR prediction")):
        fig.text(xx, 0.39, lab, fontsize=9)
    show_keys = ["truth", "h", "c", "z"]
    chk = CheckButtons(fig.add_axes([0.60, 0.06, 0.14, 0.14]),
                       ["true IF", NAMES["h"], NAMES["c"], NAMES["z"]], [True] * 4)
    fig.text(0.60, 0.205, "show", fontsize=9)
    btn = Button(fig.add_axes([0.80, 0.10, 0.12, 0.08]), "Run")
    status = fig.text(0.80, 0.06, "adjust parameters, then press Run", fontsize=9)

    def run(_=None):
        p.update(B=10 ** sliders["log_B"].val, snr=sliders["snr"].val,
                 seed=int(sliders["seed"].val), margin=sliders["margin"].val,
                 jitter=sliders["jitter"].val, sigma=sliders["sigma"].val,
                 log_lam_f=sliders["log_lam_f"].val, log_lam_c=sliders["log_lam_c"].val,
                 lp_fc_rel=sliders["lp_fc_rel"].val, f_lo=sliders["f_lo"].val,
                 env_smooth=sliders["env_smooth"].val,
                 prominence=sliders["prominence"].val, thr=sliders["thr"].val,
                 min_sep=sliders["min_sep"].val, stft_len=int(sliders["stft_len"].val),
                 source=rad_src.value_selected, hold=rad_hold.value_selected == "on",
                 cr_cross=rad_cr.value_selected, mode=rad_mode.value_selected,
                 prediction=rad_pred.value_selected == "on")
        status.set_text("running ...")
        fig.canvas.draw_idle()
        plt.pause(0.01)
        last["res"] = compute(p)
        draw(axs, last["res"], p)
        status.set_text(f"B = {p['B']:.3g} Hz, SNR = {p['snr']:.0f} dB, seed {p['seed']}")
        fig.canvas.draw_idle()

    def toggle(_label):
        p["show"].update(zip(show_keys, chk.get_status()))
        draw(axs, last["res"], p)
        fig.canvas.draw_idle()

    btn.on_clicked(run)
    chk.on_clicked(toggle)
    plt.show()


if __name__ == "__main__":
    main()
