#!/usr/bin/env python3
"""
Experiment: COMPLEX signals. Phase difference + low-pass filter vs
FSST + coarse-to-fine 2-D ridge detection.

IF: rect-PSD Gaussian process of bandwidth B, mapped so min/max IF =
-/+ if_extent * fs. Signal exp(j 2 pi int IF) + circular complex white
Gaussian noise at each SNR (same IF and noise realization across SNRs).

Output <out>.npy, shape (B, S, M, N, 2) = bandwidth x SNR x method x
simulation x [mse, trunc_mse] in Hz^2 (signed IF, wrapped error), and
<out>.json with axis values, method specs and configuration.

Examples
  python exp_complex.py --quick
  python exp_complex.py --n-sims 200 --workers 16 --out results/complex1
  python exp_complex.py --n-sims 200 --workers 16 --out results/complex1 --resume
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
    if_extent=0.4,         # IF spans [-0.4 fs, +0.4 fs]
    hop=4,                 # scoring grid (samples)
    edge=0.3,              # s excluded at both ends
    trunc_frac=0.9,
    seed=12345,
)

# M = len(METHODS). Filters: see if_lib.lowpass (fc_rel = cutoff / B,
# len = FIR taps, None -> automatic). FSST params: see if_lib.fsst_estimate.
METHODS = [
    dict(name="pd_raw",          kind="phasediff"),
    dict(name="pd_fir2B",        kind="phasediff", filter=dict(type="fir", fc_rel=2)),
    dict(name="pd_fir4B",        kind="phasediff", filter=dict(type="fir", fc_rel=4)),
    dict(name="pd_butter4B",     kind="phasediff", filter=dict(type="butter", fc_rel=4)),
    dict(name="pd_phasor_fir4B", kind="phasediff", filter=dict(type="fir", fc_rel=4),
         domain="phasor"),
    dict(name="fsst_s6",         kind="fsst", params=dict(sigma=6, lam=0.002)),
    dict(name="fsst_s10",        kind="fsst", params=dict(sigma=10, lam=0.002)),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bandwidths", type=float, nargs="+")
    p.add_argument("--snrs-db", type=float, nargs="+")
    p.add_argument("--n-sims", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results/exp_complex")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="smoke test: 2 B x 2 SNR x 2 sims")
    a = p.parse_args()
    cfg = dict(CONFIG)
    if a.quick:
        cfg.update(bandwidths=[1.0, 10.0], snrs_db=[0.0, 20.0], n_sims=2)
    for k in ("bandwidths", "snrs_db", "n_sims", "duration", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    res = L.run_experiment("complex", cfg, METHODS, a.out, a.workers, a.resume,
                           a.checkpoint_every)
    L.summarize(res, cfg, METHODS, metric=0)
    L.summarize(res, cfg, METHODS, metric=1)


if __name__ == "__main__":
    main()
