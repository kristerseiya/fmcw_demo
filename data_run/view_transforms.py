#!/usr/bin/env python3
"""
Compute and plot four time-frequency representations of any 1-D signal, and
(optionally) the representations EXPECTED from a known instantaneous frequency:

    representation  computed from the data           model from the IF
    STFT            Gaussian-window STFT (complex)   first order (Oberlin et al. eq. 6): x(t) g_hat(w - IF)
    FSST            synchrosqueezed STFT (energy)    ideal: all energy at IF(t)
    chirplet        chirplet transform (complex, 3-D) second order (Zhu et al. eq. 5, local chirp)
    FCRM            freq.-chirprate reassignment      ideal: all energy at (IF, chirp rate)

STFT and FSST are images (data | model); the 3-D arrays open in slice_viewer.py,
oriented as (chirp rate, frequency, time):
    panel 1 = time-frequency image at one chirp rate
    panel 2 = time-chirp-rate image at one frequency
    panel 3 = frequency-chirp-rate slice at one time
The viewer works in indices (i, j, k); the window title gives the conversion.
Values are in dB relative to each array's maximum (3-D viewers: --linear for raw values).

Usage
    python view_transforms.py                                    # demo signal with known IFs
    python view_transforms.py --npy sig.npy --fs 1000            # transforms only
    python view_transforms.py --npy sig.npy --fs 1000 --if-npy if.npy   # + models
    python view_transforms.py --linear                           # 3-D viewers in linear units
    python view_transforms.py --transforms stft chirplet         # only these (faster)
    python view_transforms.py --save out                         # write PNGs, no windows
--if-npy: IF in Hz, shape (N,) or (K, N) for K components, same length as the signal.
Requires chirplet_ridge.py and slice_viewer.py in the same folder.
"""
import argparse
import sys

import numpy as np
import matplotlib
if "--save" in sys.argv:                 # headless rendering; must precede the pyplot import
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import scipy.io

import tf_lib.chirplet_ridge as CR
from tf_lib.transforms import compute_transforms, model_transforms
from tf_lib.viewer import view_3d, image, overlay_if, to_db

PARAMS = dict(
    sigma=8.0,             # Gaussian window std (samples)
    hop=4,                 # frame step (samples)
    nfft=128,              # FFT size of the transforms (>= ~8 sigma)
    n_beta_in=32,          # chirp rates at which the chirplet transform is computed
    max_chirp_hz_s=None,   # chirp-rate range +/- Hz/s; None -> fs^2 / (8 sigma)
    n_omega_out=256,       # frequency bins of FSST / FCRM
    n_beta_out=64,         # chirp-rate bins of FCRM
    mode="eq9",            # FCRM frequency operator: eq9 | joint | eq6
)
DYN_RANGE_DB = 50.0        # displayed dynamic range
TRANSFORMS = ("stft", "fsst", "chirplet", "fcrm")
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- data
def load_data(path=None, idx_start=0, idx_end=1000):
    """Return (x, fs, inst_freq, phi0). inst_freq: Hz, (K, N) or None; phi0: (K,) or None."""
    if path is not None:
        if path.endswith(".h5"):
            f = h5py.File(path, 'r')
            signal = np.array(f["/channels/channel_1"])[idx_start:idx_end]
            mzi = np.array(f["/channels/channel_2"])[idx_start:idx_end]
        elif path.endswith(".mat"):
            data_path = "LNOI_EDBR_manuscript_data_modified/Figure 4/LiDAR/sig_mzi.mat"
            f = scipy.io.loadmat(path)
            signal = f["Sig"][idx_start:idx_end,0]
            mzi = f["MZI"][idx_start:idx_end,0]
        else:
            raise RuntimeError("could not open this file type")
        return np.stack([signal, mzi]), 1, None, None


    # Demo: two-component signal of Zhu et al. (eq. 3); IFs cross at t = 0.76 s
    fs = 512.0
    t = np.arange(512) / fs
    x = (np.exp(-0.03 * t) * np.sin(2 * np.pi * (-90 * t ** 2 + 220 * t))
         + (1 + 0.1 * np.cos(20 * np.pi * t)) * np.cos(2 * np.pi * (10 * np.sin(2 * np.pi * t) + 80 * t)))
    x = x + 0.2 * np.random.default_rng(0).standard_normal(t.size)
    inst_freq = np.array([220 - 180 * t, 80 + 20 * np.pi * np.cos(2 * np.pi * t)])
    phi0 = np.array([-np.pi / 2, 0.0])          # sin = cos(. - pi/2)
    return x, fs, inst_freq, phi0

# ---------------------------------------------------------------- main
def make_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", help="path to hdf5 file")
    ap.add_argument("--idx-start", help="starting index", default=0, type=int)
    ap.add_argument("--idx-end", help="ending index", default=6250, type=int)
    ap.add_argument("--transforms", nargs="+", default=list(TRANSFORMS), choices=TRANSFORMS,
                    help="which representations to compute and show (default: all)")
    ap.add_argument("--linear", action="store_true", help="3-D viewers in linear units instead of dB")
    ap.add_argument("--save", help="save figures as <prefix>_*.png instead of showing")
    return ap


def main():
    run(make_parser().parse_args())


def run(args, reassign=CR.chirplet_reassign):
    x, fs, inst_freq, phi0 = load_data(args.path, args.idx_start, args.idx_end)
    if np.ndim(x) and x.shape[0] == 2:

        for i in range(2):
            tf = compute_transforms(x[i], fs, **PARAMS, reassign=reassign, which=args.transforms)
            model = model_transforms(inst_freq, tf, phi0=phi0) if inst_freq is not None else None

            figures, db = [], not args.linear       # keep the viewer objects: their widgets need them
            images_2d = [(k, title, f_key) for k, title, f_key in
                        (("stft", "STFT |V|²", "f_in"), ("fsst", "FSST", "f_out")) if k in tf]
            if images_2d:
                rows = 2 if model else 1
                fig, ax = plt.subplots(rows, len(images_2d), figsize=(6.5 * len(images_2d), 4.3 * rows),
                                    constrained_layout=True, squeeze=False)
                for col, (k, title, f_key) in enumerate(images_2d):
                    power = np.abs(tf[k]) ** 2 if k == "stft" else tf[k]
                    image(ax[0, col], tf["t"], tf[f_key], to_db(power) if db else power, f"{title}, data (dB)")
                    if model:
                        overlay_if(ax[0, col], tf, inst_freq)
                        power = np.abs(model[k]) ** 2 if k == "stft" else model[k]
                        label = "model (first order)" if k == "stft" else "ideal"
                        image(ax[1, col], tf["t"], tf[f_key], to_db(power) if db else power, f"{title}, {label} (dB)")
                figures.append(("2d", fig))

            for k, title, f_key, c_key, label in (
                    ("chirplet", "Chirplet |C|²", "f_in", "c_in", "model (eq. 5)"),
                    ("fcrm", f"FCRM ({PARAMS['mode']})", "f_out", "c_out", "ideal")):
                if k not in tf:
                    continue
                power = np.abs(tf[k]) ** 2 if k == "chirplet" else tf[k]
                figures.append((k, view_3d(power, tf, f_key, c_key, f"{title}, data", db)))
                if model:
                    power = np.abs(model[k]) ** 2 if k == "chirplet" else model[k]
                    figures.append((f"{k}_model", view_3d(power, tf, f_key, c_key, f"{title}, {label}", db)))

    if args.save:
        for name, f in figures:
            getattr(f, "fig", f).savefig(f"{args.save}_{name}.png", dpi=110 if name == "2d" else 90)
    else:
        plt.show()            # all windows; the viewer objects stay alive until closed


if __name__ == "__main__":
    main()
