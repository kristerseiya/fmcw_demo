#!/usr/bin/env python3
"""
Interactive demo: can the SIGN of the IF be recovered from a REAL signal?

IF: rect-PSD GP mapped onto [f_center - 0.4 fs, f_center + 0.4 fs] (default
centre 0), so it changes sign. Errors are against the SIGNED true IF.

FCRM + 3-D ridge, two-sided: a real signal holds both +IF and -IF; at a sign
  change these two images cross at 0 Hz with opposite chirp rates, so a ridge
  that keeps its chirp rate continuous passes through the crossing and stays
  on one signed trajectory. Global sign: the estimate is multiplied by -1 if
  that lowers the error (oracle).
Hilbert, signed: turning points (sign changes) are detected as dips of the
  envelope of the band-passed signal ([f_lo, f_hi] rejects |IF| < f_lo); the
  folded Hilbert phase differences are multiplied by a sign sequence that
  alternates at each turning point, then low-passed. Same oracle global sign.

Panels: STFT (two-sided) + signed IF overlays | FCRM summed over chirp rate |
band-passed envelope with detected / true sign changes | signed error.
The signal is simulated at --fs (Hz); the estimators run at their default
fs = 1, so their parameters are in samples and cycles/sample (f_lo,
envelope smoothing, min separation) and the GP prior is passed as B / fs.
Results are converted back to s and Hz for display and scoring.
Adjust sliders / radio buttons, then press "Run".

  python demo_real_signed.py
  python demo_real_signed.py --B 1 --snr 10 --f-lo 0.04
  python demo_real_signed.py --save signed.png      # no window
"""
import argparse
import time
import numpy as np
import matplotlib

import tf_lib.if_lib as L

STYLE = dict(
    truth=dict(color="#FFB000", lw=6.0, alpha=0.5, solid_capstyle="round", zorder=2),
    ridge=dict(color="#1A4FD6", lw=2.0, alpha=0.9, zorder=3),
    pd=dict(color="#DC267F", lw=1.0, alpha=1.0, zorder=4),
)
LEGEND = dict(fontsize=7, loc="upper right", framealpha=0.9, facecolor="white")


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


def compute(p):
    fs, n = p["fs"], int(round(p["duration"] * p["fs"]))
    rng = np.random.default_rng(p["seed"])
    f = L.if_real(rng, n, fs, p["B"], p["f_center"], p["if_extent"])
    f_scale = p["if_extent"] * fs / 2
    s = np.cos(L.phase_from_if(f, fs, rng.uniform(0, 2 * np.pi)))
    x = L.add_noise(s, L.unit_noise(rng, n, False), p["snr"])
    B_n, f_scale_n = p["B"] / fs, f_scale / fs          # estimators at fs = 1: cycles/sample
    filt = None if p["lp_type"] == "none" else dict(
        type=p["lp_type"], fc_rel=p["lp_fc_rel"], len=(p["lp_len"] or None))
    tp = dict(f_lo=p["f_lo"], order=p["bp_order"], env_smooth=p["env_smooth"],
              prominence=p["prominence"], thr=p["thr"], min_sep=p["min_sep"])
    t0 = time.perf_counter()
    th, fh, (te, env, idx) = L.signed_hilbert_estimate(x, B_n, filt,
                                                       p["sign_before_filter"], tp)
    t1 = time.perf_counter()
    tc, fc, cr, tfr = L.fcrm_estimate(x, B_n, f_scale_n, sigma=p["sigma"],
                                      lam_f=10 ** p["log_lam_f"], lam_c=10 ** p["log_lam_c"],
                                      mode=p["mode"], prediction=p["prediction"],
                                      onesided=False, n_omega_out=512, return_tfr=True)
    t2 = time.perf_counter()
    th, fh, te = th / fs, fh * fs, te / fs              # samples -> s, cycles/sample -> Hz
    tc, fc = tc / fs, fc * fs
    sg_h, fh = L.best_global_sign(th, fh, f, fs, edge=p["edge"])
    sg_c, fc = L.best_global_sign(tc, fc, f, fs, edge=p["edge"])
    res = dict(t=np.arange(n) / fs, f=f, x=x, th=th, fh=fh, tc=tc, fc=fc, tfr=tfr,
               te=te, env=env, idx=idx, sg_h=sg_h, sg_c=sg_c,
               zc=np.nonzero(np.diff(np.sign(f)) != 0)[0], time_h=t1 - t0, time_c=t2 - t1)
    for k, (tt, fe) in (("h", (th, fh)), ("c", (tc, fc))):
        tg, e = L.errors(tt, fe, f, fs, edge=p["edge"])          # signed, not wrapped
        res["e_" + k] = (tg, e)
        res["m_" + k] = np.sqrt(L.metrics(e))
        res["ok_" + k] = sign_agreement(tt, fe, f, fs, p["edge"])
    return res


def draw(axs, res, p):
    a0, a1, a2, a3 = axs
    for a in axs:
        a.clear()
    fs, vis = p["fs"], p["show"]
    ts, fr, S = L.spectrogram(res["x"], fs, L=p["stft_len"], hop=max(1, p["stft_len"] // 8),
                              twosided=True)
    show_image(a0, ts, fr, S, -p["dyn_range"], 0, p["cmap"])
    if vis["truth"]:
        a0.plot(res["t"], res["f"], label="true IF (signed)", **STYLE["truth"])
    if vis["ridge"]:
        a0.plot(res["tc"], res["fc"], label=f"FCRM two-sided (global sign {res['sg_c']:+d})",
                **STYLE["ridge"])
    if vis["pd"]:
        a0.plot(res["th"], res["fh"], label=f"Hilbert signed (global sign {res['sg_h']:+d})",
                **STYLE["pd"])
    a0.set_ylim(-fs / 2, fs / 2); a0.set_xlim(0, p["duration"])
    a0.axhline(0, color="k", lw=0.5, alpha=0.5)
    a0.set_xlabel("time (s)"); a0.set_ylabel("frequency (Hz)")
    a0.set_title(f"STFT (Hann {p['stft_len']}), B = {p['B']:.2g} Hz, SNR = {p['snr']:.0f} dB")
    if any(vis.values()):
        a0.legend(**LEGEND)

    tfr = res["tfr"]
    Rs = tfr["R"].sum(-1)
    fo = tfr["omega_out"] * fs / (2 * np.pi)
    show_image(a1, tfr["t"] / fs, fo, 10 * np.log10(Rs.T / Rs.max() + 1e-6),
               -p["dyn_range"], 0, p["cmap"])
    if vis["truth"]:
        a1.plot(res["t"], res["f"], **STYLE["truth"])
    if vis["ridge"]:
        a1.plot(res["tc"], res["fc"], **STYLE["ridge"])
    a1.set_ylim(-fs / 2, fs / 2); a1.set_xlim(0, p["duration"])
    a1.set_title(f"FCRM two-sided ({p['mode']}, σ = {p['sigma']:.1f}); "
                 f"λf = 10^{p['log_lam_f']:.1f}, λc = 10^{p['log_lam_c']:.1f}", fontsize=9)
    a1.set_xlabel("time (s)")

    a2.plot(res["te"], res["env"], color="#555555", lw=0.8, label="band-passed envelope / median")
    a2.axhline(p["thr"], color="k", ls="--", lw=0.8, label=f"level cap {p['thr']:.2f}")
    for i, k in enumerate(res["zc"]):
        a2.axvline(k / fs, color=STYLE["truth"]["color"], lw=3, alpha=0.6,
                   label="true sign change" if i == 0 else None)
    a2.plot(res["idx"] / fs, res["env"][res["idx"]], "x", color=STYLE["pd"]["color"], ms=8,
            mew=2, label=f"detected ({len(res['idx'])} vs {len(res['zc'])} true)")
    a2.set_ylim(0, max(2.5, 1.1 * np.percentile(res["env"], 99)))
    a2.set_xlim(0, p["duration"])
    a2.set_title(f"turning points: band-pass [{p['f_lo']:.3g}, 0.45] cyc/sample "
                 f"([{p['f_lo'] * fs:.0f}, {0.45 * fs:.0f}] Hz), "
                 f"smooth {p['env_smooth']:.0f} samples, prominence {p['prominence']:.2f}",
                 fontsize=9)
    a2.set_xlabel("time (s)"); a2.legend(**LEGEND)

    for k, sk, name in (("h", "pd", "Hilbert"), ("c", "ridge", "FCRM")):
        tg, e = res["e_" + k]
        m = res["m_" + k]
        a3.plot(tg, e, color=STYLE[sk]["color"], lw=1.0, zorder=STYLE[sk]["zorder"],
                label=f"{name}: RMSE {m[0]:.1f}, trunc {m[1]:.1f} Hz, "
                      f"sign correct {res['ok_' + k]:.0%}")
    a3.axhline(0, color="gray", lw=0.6); a3.grid(alpha=0.25)
    lim = 5 * max(res["m_h"][1], res["m_c"][1], 2.0)
    a3.set_ylim(-min(lim, fs), min(lim, fs))
    a3.set_title(f"signed error (estimate - IF), {p['edge']} s ends excluded  "
                 f"[Hilbert {res['time_h'] * 1e3:.0f} ms, FCRM {res['time_c'] * 1e3:.0f} ms]",
                 fontsize=9)
    a3.set_xlabel("time (s)"); a3.set_ylabel("Hz"); a3.legend(**LEGEND)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=float, default=1.0, help="GP bandwidth (Hz)")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    # FCRM
    ap.add_argument("--sigma", type=float, default=8.0, help="FCRM Gaussian std (samples)")
    ap.add_argument("--lam-f", type=float, default=0.05)
    ap.add_argument("--lam-c", type=float, default=0.3)
    ap.add_argument("--mode", default="eq9", choices=["eq9", "joint", "eq6"])
    ap.add_argument("--no-prediction", action="store_true")
    # Hilbert IF filter
    ap.add_argument("--lp-type", default="fir", choices=["none", "fir", "ma", "butter"])
    ap.add_argument("--lp-len", type=int, default=0, help="0 = auto from cutoff")
    ap.add_argument("--lp-fc-rel", type=float, default=4.0, help="cutoff = this * B")
    ap.add_argument("--sign-after-filter", action="store_true",
                    help="filter |IF| first, then apply the sign sequence")
    # turning-point detection (estimator units: cycles/sample, samples)
    ap.add_argument("--f-lo", type=float, default=0.05,
                    help="band-pass low edge (cycles/sample; 0.05 = 50 Hz at fs = 1 kHz)")
    ap.add_argument("--bp-order", type=int, default=4)
    ap.add_argument("--env-smooth", type=float, default=15.0,
                    help="envelope moving-average length (samples)")
    ap.add_argument("--prominence", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.6, help="dip level cap (x median)")
    ap.add_argument("--min-sep", type=float, default=30.0,
                    help="minimum separation of turning points (samples)")
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

    p = dict(B=a.B, snr=a.snr, seed=a.seed, sigma=a.sigma, log_lam_f=np.log10(a.lam_f),
             log_lam_c=np.log10(a.lam_c), mode=a.mode, prediction=not a.no_prediction,
             lp_type=a.lp_type, lp_len=a.lp_len, lp_fc_rel=a.lp_fc_rel,
             sign_before_filter=not a.sign_after_filter, f_lo=a.f_lo, bp_order=a.bp_order,
             env_smooth=a.env_smooth, prominence=a.prominence, thr=a.thr,
             min_sep=a.min_sep, fs=a.fs, duration=a.duration, f_center=a.f_center,
             if_extent=a.if_extent, edge=a.edge, stft_len=a.stft_len, cmap=a.cmap,
             dyn_range=a.dyn_range, show=dict(truth=True, pd=True, ridge=True))

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
        ("sigma", "FCRM σ (samples)", 3, 20, p["sigma"], 0.5),
        ("log_lam_f", "log10 λ freq", -4, 1, p["log_lam_f"], 0.1),
        ("log_lam_c", "log10 λ chirp", -4, 1, p["log_lam_c"], 0.1),
        ("lp_len", "LP len (0=auto)", 0, 2001, p["lp_len"], 2),
        ("lp_fc_rel", "LP cutoff / B", 0.5, 12, p["lp_fc_rel"], 0.5),
        ("f_lo", "band-pass f_lo (cyc/sample)", 0.005, 0.2, p["f_lo"], 0.001),
        ("bp_order", "band-pass order", 1, 8, p["bp_order"], 1),
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
    lp_opts, mode_opts, onoff = ["none", "fir", "ma", "butter"], ["eq9", "joint", "eq6"], ["on", "off"]
    order_opts = ["before filter", "after filter"]
    rad_lp = RadioButtons(fig.add_axes([0.60, 0.25, 0.08, 0.13]), lp_opts,
                          active=lp_opts.index(p["lp_type"]))
    rad_sign = RadioButtons(fig.add_axes([0.70, 0.29, 0.10, 0.09]), order_opts,
                            active=0 if p["sign_before_filter"] else 1)
    rad_mode = RadioButtons(fig.add_axes([0.82, 0.27, 0.07, 0.11]), mode_opts,
                            active=mode_opts.index(p["mode"]))
    rad_pred = RadioButtons(fig.add_axes([0.91, 0.29, 0.07, 0.09]), onoff,
                            active=0 if p["prediction"] else 1)
    for xx, lab in ((0.60, "Hilbert LP filter"), (0.70, "apply sign"),
                    (0.82, "FCRM operator"), (0.91, "CR prediction")):
        fig.text(xx, 0.39, lab, fontsize=9)
    show_keys = ["truth", "pd", "ridge"]
    chk = CheckButtons(fig.add_axes([0.60, 0.08, 0.12, 0.11]),
                       ["true IF", "Hilbert", "FCRM ridge"], [True, True, True])
    fig.text(0.60, 0.195, "show", fontsize=9)
    btn = Button(fig.add_axes([0.80, 0.10, 0.12, 0.08]), "Run")
    status = fig.text(0.80, 0.06, "adjust parameters, then press Run", fontsize=9)

    def run(_=None):
        p.update(B=10 ** sliders["log_B"].val, snr=sliders["snr"].val,
                 seed=int(sliders["seed"].val), sigma=sliders["sigma"].val,
                 log_lam_f=sliders["log_lam_f"].val, log_lam_c=sliders["log_lam_c"].val,
                 lp_len=int(sliders["lp_len"].val), lp_fc_rel=sliders["lp_fc_rel"].val,
                 f_lo=sliders["f_lo"].val, bp_order=int(sliders["bp_order"].val),
                 env_smooth=sliders["env_smooth"].val,
                 prominence=sliders["prominence"].val, thr=sliders["thr"].val,
                 min_sep=sliders["min_sep"].val, stft_len=int(sliders["stft_len"].val),
                 lp_type=rad_lp.value_selected, mode=rad_mode.value_selected,
                 prediction=rad_pred.value_selected == "on",
                 sign_before_filter=rad_sign.value_selected == "before filter")
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
