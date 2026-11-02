#!/usr/bin/env python3
"""
Experiment: REAL signals. Hilbert transform + phase difference + low-pass
filter vs FCRM (chirplet + frequency-chirprate reassignment) + coarse-to-fine
3-D ridge detection.

IF: rect-PSD Gaussian process of bandwidth B, mapped (as in the complex
case) so min/max IF = f_center -/+ if_extent * fs; default f_center = 0, so
the IF spans [-0.4 fs, 0.4 fs] and crosses zero. Signal cos(2 pi int IF) +
white Gaussian noise at each SNR (same IF and noise across SNRs).
A real signal cannot reveal the sign of the IF, so errors are
estimate - |IF|: FCRM works on positive frequencies (its estimate is |.|),
the Hilbert phase differences are wrapped (folded) into [0, fs/2].

Output <out>.npy, shape (B, S, M, N, 2) = bandwidth x SNR x method x
simulation x [mse, trunc_mse] in Hz^2, and <out>.json with axis values,
method specs and configuration.

Examples
  python exp_real.py --quick
  python exp_real.py --n-sims 200 --workers 16 --out results/real1
  python exp_real.py --n-sims 200 --workers 16 --out results/real1 --resume
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import tf_lib.if_lib as L

CONFIG = dict(
    bandwidths=[0.5, 1, 2, 5, 10, 20],   # GP bandwidth / top frequency (Hz)
    snrs_db=[-5, 0, 10, 20, 30],
    n_sims=100,
    fs=1000.0,
    duration=4.0,          # s
    f_center=0.0,          # centre of the IF range (Hz)
    if_extent=0.4,         # IF spans f_center +/- if_extent * fs
    hop=4,                 # scoring grid (samples)
    edge=0.3,              # s excluded at both ends
    trunc_frac=0.9,
    seed=12345,
)

# M = len(METHODS). Filters: see if_lib.lowpass. fold="pre": each Hilbert
# phase difference is wrapped into [0, fs/2] before filtering ("post": the
# filtered estimate is folded instead). FCRM params: see
# if_lib.fcrm_estimate (sigma = window std in samples, lam_f / lam_c =
# frequency / chirp-rate penalties, mode = "eq9" | "joint" | "eq6").
METHODS = [
    dict(name="hil_raw",          kind="phasediff", analytic=True, fold="pre"),
    dict(name="hil_fir2B",        kind="phasediff", analytic=True, fold="pre",
         filter=dict(type="fir", fc_rel=2)),
    dict(name="hil_fir4B",        kind="phasediff", analytic=True, fold="pre",
         filter=dict(type="fir", fc_rel=4)),
    dict(name="hil_butter4B",     kind="phasediff", analytic=True, fold="pre",
         filter=dict(type="butter", fc_rel=4)),
    dict(name="hil_phasor_fir4B", kind="phasediff", analytic=True, fold="pre",
         filter=dict(type="fir", fc_rel=4), domain="phasor"),
    dict(name="fcrm_s6",          kind="fcrm", params=dict(sigma=6, lam_f=0.05, lam_c=0.3)),
    dict(name="fcrm_s10",         kind="fcrm", params=dict(sigma=10, lam_f=0.05, lam_c=0.3)),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bandwidths", type=float, nargs="+")
    p.add_argument("--snrs-db", type=float, nargs="+")
    p.add_argument("--n-sims", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--f-center", type=float)
    p.add_argument("--if-extent", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results/exp_real")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="smoke test: 2 B x 2 SNR x 2 sims")
    a = p.parse_args()
    cfg = dict(CONFIG)
    if a.quick:
        cfg.update(bandwidths=[1.0, 10.0], snrs_db=[0.0, 20.0], n_sims=2)
    for k in ("bandwidths", "snrs_db", "n_sims", "duration", "f_center", "if_extent", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    res = L.run_experiment("real", cfg, METHODS, a.out, a.workers, a.resume,
                           a.checkpoint_every)
    L.summarize(res, cfg, METHODS, metric=0)
    L.summarize(res, cfg, METHODS, metric=1)


if __name__ == "__main__":
    main()
