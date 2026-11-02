#!/usr/bin/env python3
"""
Interactive demo: COMPLEX signal, phase difference + low-pass filter vs
FSST + coarse-to-fine 2-D ridge.

Panels: spectrogram with true IF and both estimates overlaid | FSST map
with its ridge | error vs time (RMSE and truncated RMSE in the legend).
Adjust the sliders / radio buttons, then press "Run".

  python demo_complex.py                       # interactive window
  python demo_complex.py --B 5 --snr 0         # start from other values
  python demo_complex.py --save demo.png       # no window: render once, save
"""
import argparse
import time
import numpy as np
import matplotlib

import tf_lib.if_lib as L


# Overlay styles: light grayscale background (energy = dark), colorblind-safe
# line colors, truth as a wide translucent band drawn underneath the estimates,
# the thin phase-difference line drawn over the wider ridge line so both stay
# visible where they agree. The image is lightened (alpha) so lines stand out.
STYLE = dict(
    truth=dict(color="#FFB000", lw=6.0, alpha=0.5, solid_capstyle="round", zorder=2),
    ridge=dict(color="#1A4FD6", lw=2.0, alpha=0.9, zorder=3),
    pd=dict(color="#DC267F", lw=1.0, alpha=1.0, zorder=4),   # thin, on top of the ridge
    signed=dict(color="#444444", lw=1.0, ls=":", zorder=2),
)
LEGEND = dict(fontsize=7, loc="upper right", framealpha=0.9, facecolor="white")


def show_image(ax, t, f, Z, vmin, vmax, cmap="Greys", alpha=0.8):
    """Fast image of Z (len(f) x len(t)) on regular axes t, f."""
    dt = (t[1] - t[0]) if len(t) > 1 else 1.0
    dfr = (f[1] - f[0]) if len(f) > 1 else 1.0
    ax.imshow(Z, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=vmin, vmax=vmax, alpha=alpha,
              extent=[t[0] - dt / 2, t[-1] + dt / 2, f[0] - dfr / 2, f[-1] + dfr / 2])


def compute(p):
    fs, n = p["fs"], int(round(p["duration"] * p["fs"]))
    rng = np.random.default_rng(p["seed"])
    f = L.if_complex(rng, n, fs, p["B"], p["if_extent"])
    s = np.exp(1j * L.phase_from_if(f, fs, rng.uniform(0, 2 * np.pi)))
    x = L.add_noise(s, L.unit_noise(rng, n, True), p["snr"])
    f_scale = p["if_extent"] * fs / 2
    filt = None if p["lp_type"] == "none" else dict(
        type=p["lp_type"], fc_rel=p["lp_fc_rel"], len=(p["lp_len"] or None))
    t0 = time.perf_counter()
    tp, fp = L.phasediff_estimate(x, fs, p["B"], filt, p["domain"])
    t1 = time.perf_counter()
    ts, fsst, tfr = L.fsst_estimate(x, fs, p["B"], f_scale, sigma=p["sigma"],
                                    lam=10 ** p["log_lam"], return_tfr=True)
    t2 = time.perf_counter()
    res = dict(t=np.arange(n) / fs, f=f, x=x, tp=tp, fp=fp, ts=ts, fs_est=fsst, tfr=tfr,
               time_pd=t1 - t0, time_sst=t2 - t1)
    for k, (te, fe) in (("pd", (tp, fp)), ("sst", (ts, fsst))):
        tg, e = L.errors(te, fe, f, fs, edge=p["edge"], wrap=True)
        res["e_" + k] = (tg, e)
        res["m_" + k] = np.sqrt(L.metrics(e))
    return res


def draw(axs, res, p):
    a0, a1, a2 = axs
    for a in axs:
        a.clear()
    fs, vis = p["fs"], p["show"]
    ts, fr, S = L.spectrogram(res["x"], fs, L=p["stft_len"], hop=max(1, p["stft_len"] // 8),
                              twosided=True)
    show_image(a0, ts, fr, S, -p["dyn_range"], 0, p["cmap"])
    if vis["truth"]:
        a0.plot(res["t"], res["f"], label="true IF", **STYLE["truth"])
    if vis["pd"]:
        a0.plot(res["tp"], res["fp"], label=f"phase diff ({p['lp_type']}, {p['domain']})",
                **STYLE["pd"])
    if vis["ridge"]:
        a0.plot(res["ts"], res["fs_est"], label="FSST + 2-D ridge", **STYLE["ridge"])
    a0.set_ylim(-fs / 2, fs / 2)
    a0.set_xlim(0, p["duration"])
    a0.set_xlabel("time (s)"); a0.set_ylabel("frequency (Hz)")
    a0.set_title(f"STFT (Hann {p['stft_len']}), B = {p['B']:.2g} Hz, SNR = {p['snr']:.0f} dB")
    if any(vis.values()):
        a0.legend(**LEGEND)

    tfr = res["tfr"]
    Rm = tfr["R"][:, :, 0]
    fo = tfr["omega_out"] * fs / (2 * np.pi)
    show_image(a1, tfr["t"] / fs, fo, 10 * np.log10(Rm.T / Rm.max() + 1e-6),
               -p["dyn_range"], 0, p["cmap"])
    if vis["truth"]:
        a1.plot(res["t"], res["f"], **STYLE["truth"])
    if vis["ridge"]:
        a1.plot(res["ts"], res["fs_est"], **STYLE["ridge"])
    a1.set_ylim(-fs / 2, fs / 2)
    a1.set_title(f"FSST (σ = {p['sigma']:.1f}), ridge λ = 10^{p['log_lam']:.1f}")
    a1.set_xlim(0, p["duration"])
    a1.set_xlabel("time (s)")

    for k, sk, name in (("pd", "pd", "phase diff"), ("sst", "ridge", "FSST")):
        tg, e = res["e_" + k]
        m = res["m_" + k]
        a2.plot(tg, e, color=STYLE[sk]["color"], lw=1.0, zorder=STYLE[sk]["zorder"],
                label=f"{name}: RMSE {m[0]:.2f}, trunc {m[1]:.2f} Hz")
    lim = 5 * max(res["m_pd"][1], res["m_sst"][1], 1.0)
    a2.set_ylim(-lim, lim)
    a2.axhline(0, color="gray", lw=0.6)
    a2.grid(alpha=0.25)
    a2.set_title(f"error (wrapped), {p['edge']} s ends excluded  "
                 f"[pd {res['time_pd'] * 1e3:.0f} ms, FSST {res['time_sst'] * 1e3:.0f} ms]")
    a2.set_xlabel("time (s)"); a2.set_ylabel("Hz"); a2.legend(**LEGEND)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=float, default=2.0, help="GP bandwidth (Hz)")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=6.0, help="FSST Gaussian std (samples)")
    ap.add_argument("--lam", type=float, default=0.002, help="FSST ridge penalty")
    ap.add_argument("--lp-type", default="fir", choices=["none", "fir", "ma", "butter"])
    ap.add_argument("--lp-len", type=int, default=0, help="FIR/MA length; 0 = auto from cutoff (FIR 4 fs/fc, MA 0.443 fs/fc)")
    ap.add_argument("--lp-fc-rel", type=float, default=4.0, help="cutoff = this * B")
    ap.add_argument("--domain", default="freq", choices=["freq", "phasor"])
    ap.add_argument("--stft-len", type=int, default=128, help="display STFT window")
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--if-extent", type=float, default=0.4)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--cmap", default="Greys",
                    help="colormap for the time-frequency images (e.g. Greys, gray_r, bone_r)")
    ap.add_argument("--dyn-range", type=float, default=40.0, help="displayed dB range")
    ap.add_argument("--save", help="render once to this image file, no window")
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

    p = dict(B=a.B, snr=a.snr, seed=a.seed, sigma=a.sigma, log_lam=np.log10(a.lam),
             lp_type=a.lp_type, lp_len=a.lp_len, lp_fc_rel=a.lp_fc_rel, domain=a.domain,
             stft_len=a.stft_len, fs=a.fs, duration=a.duration, if_extent=a.if_extent,
             edge=a.edge, cmap=a.cmap, dyn_range=a.dyn_range,
             show=dict(truth=True, pd=True, ridge=True))

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.95, bottom=0.36,
                          hspace=0.35, wspace=0.15)
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    last = {"res": compute(p)}
    draw(axs, last["res"], p)
    if a.save:
        fig.savefig(a.save, dpi=100)
        print(f"saved {a.save}")
        return

    # ---- controls
    specs = [  # key, label, min, max, init, step
        ("log_B", "log10 B (Hz)", -1.0, 1.5, np.log10(p["B"]), None),
        ("snr", "SNR (dB)", -10, 40, p["snr"], 1),
        ("seed", "seed", 0, 99, p["seed"], 1),
        ("sigma", "FSST σ (samples)", 2, 24, p["sigma"], 0.5),
        ("log_lam", "log10 ridge λ", -5, 1, p["log_lam"], 0.1),
        ("lp_len", "LP len (0=auto from cutoff)", 0, 2001, p["lp_len"], 2),
        ("lp_fc_rel", "LP cutoff / B", 0.5, 12, p["lp_fc_rel"], 0.5),
        ("stft_len", "display STFT len", 16, 512, p["stft_len"], 16),
    ]
    sliders = {}
    for i, (k, lab, lo, hi, v, st) in enumerate(specs):
        ax = fig.add_axes([0.12, 0.29 - i * 0.033, 0.42, 0.022])
        sliders[k] = Slider(ax, lab, lo, hi, valinit=v, valstep=st)
    rad_lp = RadioButtons(fig.add_axes([0.63, 0.12, 0.09, 0.15]),
                          ["none", "fir", "ma", "butter"],
                          active=["none", "fir", "ma", "butter"].index(p["lp_type"]))
    rad_dom = RadioButtons(fig.add_axes([0.74, 0.17, 0.09, 0.10]), ["freq", "phasor"],
                           active=["freq", "phasor"].index(p["domain"]))
    fig.text(0.63, 0.28, "LP filter type", fontsize=9)
    fig.text(0.74, 0.28, "filter domain", fontsize=9)
    btn = Button(fig.add_axes([0.86, 0.17, 0.1, 0.07]), "Run")
    status = fig.text(0.63, 0.05, "", fontsize=9)

    def run(_=None):
        p.update(B=10 ** sliders["log_B"].val, snr=sliders["snr"].val,
                 seed=int(sliders["seed"].val), sigma=sliders["sigma"].val,
                 log_lam=sliders["log_lam"].val, lp_len=int(sliders["lp_len"].val),
                 lp_fc_rel=sliders["lp_fc_rel"].val, stft_len=int(sliders["stft_len"].val),
                 lp_type=rad_lp.value_selected, domain=rad_dom.value_selected)
        status.set_text("running ...")
        fig.canvas.draw_idle()
        plt.pause(0.01)
        last["res"] = compute(p)
        draw(axs, last["res"], p)
        status.set_text(f"B = {p['B']:.3g} Hz, SNR = {p['snr']:.0f} dB, seed {p['seed']}")
        fig.canvas.draw_idle()

    show_keys = ["truth", "pd", "ridge"]
    chk = CheckButtons(fig.add_axes([0.86, 0.03, 0.12, 0.11]), ["true IF", "phase diff", "FSST ridge"],
                       [p["show"][k] for k in show_keys])
    fig.text(0.86, 0.145, "show", fontsize=9)

    def toggle(_label):
        p["show"].update(zip(show_keys, chk.get_status()))
        draw(axs, last["res"], p)
        fig.canvas.draw_idle()

    chk.on_clicked(toggle)
    btn.on_clicked(run)
    status.set_text("adjust parameters, then press Run")
    plt.show()


if __name__ == "__main__":
    main()
