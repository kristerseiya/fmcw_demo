"""
run.py -- load data, run the algorithm on overlapping windows in parallel,
save the stacked outputs to results_<date>_<time>.npz next to this file.

Each run of process_segment sees x[idx : idx + WIN_BLOCKS*L, :] and the next
run starts HOP_BLOCKS*L rows later (L = meta["L"]).

load_data() returns (x, meta, prev). If prev is not None it is a dict of
results of a previous run, each value an array with one row per window (axis
0); window j additionally gets
process_segment(..., prev_result={key: prev[key][j] for key in prev}).

Usage:
    python run.py                            # parallel, all CPUs
    python run.py --workers 1 --max-windows 3   # serial quick test (breakpoints work)
    python run.py --on-error skip            # long runs: don't stop on a bad window
    python run.py --start 1000000 --stop 2000000 --out test.npz
"""
import argparse
import os
import time

import numpy as np
import h5py
import scipy.io
from scipy.fft import fftfreq
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fmcw_aux import construct_fractional_D_with_time, PicklableFactorized, ModuFuncSolver

import algorithm
from parallel_runner import run_windows, save_results

from path_variables import PATH_TO_0821_LASER2_TRIANGULAR, PATH_TO_0821_LASER2_SINUSOIDAL, PATH_TO_LNOI_EDBR_TRIANGULAR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/sutakuri/projects/fmcw/chirp_above_nyquist/data"

# Window layout, in units of L.
WIN_BLOCKS = 3   # each run sees 3*L rows
HOP_BLOCKS = 1   # consecutive runs are shifted by 1*L rows

def get_algo_metadata(config, args):

    if args.algo == "resample":
        config["method"] = "resample"

    elif args.algo in ["if", "mf", "bmf", "bp", "bbp"]:
        
        # if estimation method
        if args.ref_if_est == "hilbert":

            config["ref_if_est"] = "hilbert"
            config["ref_if_est_params"] = dict(fs=1000,
                                                filt=None,
                                                tp_params=dict(loc=config["ref_if_zerocrossings"],
                                                                f_lo=20),
                                                )
            if args.algo == "if":
                config["ref_if_smooth_length"] = 0
            else:
                config["ref_if_smooth_length"] = 100//args.downsample

        elif args.ref_if_est == "fcrm":

            config["ref_if_est"] = "fcrm"
            config["ref_if_est_params"] = dict(fs=1000, 
                                            lam_k=0.01 if config["src"].endswith("triangular") else 0.1, 
                                            lam_b=0.05 if config["src"].endswith("triangular") else 0.5,
                                            zero_crossings=config["ref_if_zerocrossings"] )
            if args.algo == "if":
                config["ref_if_smooth_length"] = 0
            else:
                config["ref_if_smooth_length"] = 2

        else:
            raise ValueError(f"did not recognize {args.if_est:s} for if estimation type")

        if args.col_if_est == "none":
            config["col_if_est"] = "none"

        elif args.col_if_est == "hilbert":

            config["col_if_est"] = "hilbert"
            config["col_if_est_params"] = dict(fs=1000,
                                                filt=None,
                                                tp_params=dict(loc=config["col_if_zerocrossings"],
                                                                f_lo=20),
                                                )
            if args.algo == "if":
                config["col_if_smooth_length"] = 0
            else:
                config["col_if_smooth_length"] = 100//args.downsample

        elif args.col_if_est == "fcrm":

            config["col_if_est"] = "fcrm"
            config["col_if_est_params"] = dict(fs=1000, 
                                            lam_k=0.01 if config["src"].endswith("triangular") else 0.1, 
                                            lam_b=0.05 if config["src"].endswith("triangular") else 0.5,
                                                zero_crossings=config["col_if_zerocrossings"] )
            if args.algo == "if":
                config["col_if_smooth_length"] = 0
            else:
                config["col_if_smooth_length"] = 2

        else:
            raise ValueError(f"did not recognize {args.if_est:s} for if estimation type")

        config["modf_smooth_length"] = 2//args.downsample

        # (transmitted) if estimation method
        config["inv"] = args.inverse
        del_t = config["del_t"]
        L = config["L"]

        if config["col_if_est"] == "none":
            modufuncsolver = ModuFuncSolver(L*3, 1/del_t, 
                                            [config["aux_delay"],], period=3*L*del_t, 
                                            method=config["inv"], verbose=True)
        else:
            modufuncsolver = ModuFuncSolver(L*3, 1/del_t,
                                            [config["aux_delay"], config["col_delay"]], period=3*L*del_t, 
                                            method=config["inv"], verbose=True)

        config["modfsolver"] = modufuncsolver

        min_distance, max_distance = config["min_distance"], config["max_distance"]

        config["method"] = args.algo
        
        if config["method"] == "mf":

            config["algo_params"] = dict(t_low=min_distance*2/3e8, t_high=max_distance*2/3e8)
            
        elif config["method"] == "bmf":

            config["algo_params"] = dict(window_len=1000, stride=500, 
                                        window="blackmanharris",  
                                        t_low=min_distance*2/3e8, t_high=max_distance*2/3e8)

        elif config["method"] == "bp":

            config["algo_params"] = dict(lambd=1, rho=100, n_iter=1000, tol=3e-7, 
                                       t_low=min_distance*2/3e8, t_high=max_distance*2/3e8)
            
        elif config["method"] == "bbp":

            config["algo_params"] = dict(lambd=1, window_len=1000, stride=500, rho=100, 
                                        window="blackmanharris", n_iter=100, tol=3e-7, 
                                        t_low=min_distance*2/3e8, t_high=max_distance*2/3e8)

        if config["method"] != "if":
            config["subtract_collimator"] = args.subtract_collimator
            config["find_peaks_params"] = dict(distance=10)
            config["top_k"] = 10

    
    else:
        raise ValueError(f"did not recognize {args.algo:s} for algorithm type")

    return config

def _turning_point_extension(tpidx, uncertainty, L):
    tp_indices = []
    for i in range(-(tpidx)//(L//2)+1, (3*L-tpidx)//(L//2)+1):
        tp_indices.append([ max(0,tpidx+i*(L//2)-uncertainty), min(3*L,tpidx+i*(L//2)+uncertainty) ])
    return tp_indices

def load_data(args):
    """
    Return (x, meta, prev):
      x    : the long 1d array
      meta : dict of parameters describing x; every call of
             process_segment receives it as `meta`.
      prev : None (no previous results), or a dict {key: array} of results of
             a previous run; every array has M rows (axis 0), M = number of
             windows in rows [--start, --stop) (row j = window j, an error is
             raised otherwise). Window j receives
             prev_result={key: array[j]} in process_segment.
    """
    # ---- replace with your data, e.g.
    #   d = np.load("signal.npz"); x = d["x"]; meta = dict(fs=float(d["fs"]), ...)
    """
    # Example 
    meta = dict(
        fs=1.0,        # sampling rate
        f0=0.05,       # start frequency
        k=2e-8,        # chirp rate
        snr_db=20.0,
    )
    t = np.arange(2_000_000) / meta["fs"]
    noise_std = 10 ** (-meta["snr_db"] / 20)
    x = np.cos(2 * np.pi * (meta["f0"] * t + 0.5 * meta["k"] * t**2))
    x = x + noise_std * np.random.randn(t.size)
    return x, meta
    """
    if args.data == "0821_laser2_triangular":
        f = h5py.File(PATH_TO_0821_LASER2_TRIANGULAR, 'r')
        signal = np.array(f["/channels/channel_1"])[::args.downsample]
        mzi = np.array(f["/channels/channel_2"])[::args.downsample]
        t = -0.01 + np.arange(len(signal))*3.2000000000000005e-09*args.downsample
        config = dict(del_t=t[1]-t[0],
                    aux_delay=40/3e8,
                    col_delay = np.arcsin(np.sin(np.pi*50e3*40/3e8)*0.95)/np.pi/50e3,
                    T=20e-6,
                    L=6250//args.downsample,
                    B=3.35e9,
                    min_distance=args.min_distance if args.min_distance is not None else 0,
                    max_distance=args.max_distance if args.max_distance is not None else 40,
                    target_min_distance=args.target_min_distance if args.target_min_distance is not None else 21,
                    target_max_distance=args.target_max_distance if args.target_max_distance is not None else 35,
                    cbf_indices= [[6750//args.downsample, 9250//args.downsample],[9750//args.downsample, 12250//args.downsample]],
                    ref_if_zerocrossings = _turning_point_extension( (9450-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    col_if_zerocrossings = _turning_point_extension( (9450-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    uprate=10*args.downsample,
                    downsample=args.downsample,
                    )

    elif args.data == "0821_laser2_sinusoidal":
        f = h5py.File(PATH_TO_0821_LASER2_SINUSOIDAL, 'r')
        signal = np.array(f["/channels/channel_1"])[::args.downsample]
        mzi = np.array(f["/channels/channel_2"])[::args.downsample]
        t = -0.01 + np.arange(40000000)*3.2000000000000005e-09
        config = dict(del_t=t[1]-t[0],
                    aux_delay=40/3e8,
                    col_delay = np.arcsin(np.sin(np.pi*50e3*40/3e8)*0.95)/np.pi/50e3,
                    T=20e-6,
                    L=6250,
                    B=3.35e9,
                    min_distance=args.min_distance if args.min_distance is not None else 0,
                    max_distance=args.max_distance if args.max_distance is not None else 40,
                    target_min_distance=args.target_min_distance if args.target_min_distance is not None else 21,
                    target_max_distance=args.target_max_distance if args.target_max_distance is not None else 35,
                    cbf_indices= [[6750//args.downsample, 9250//args.downsample],[9750//args.downsample, 12250//args.downsample]],
                    ref_if_zerocrossings = _turning_point_extension( (10540-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    col_if_zerocrossings = _turning_point_extension( (10540-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    uprate=10*args.downsample,
                    downsample=args.downsample,
                    )

    elif args.data == "lnoi_edbr_triangular":
        f = scipy.io.loadmat(PATH_TO_LNOI_EDBR_TRIANGULAR)
        signal = f["Sig"][::args.downsample,0]
        mzi = f["MZI"][::args.downsample,0]
        t = f["t"][::args.downsample,0]
        config = dict(del_t=t[1]-t[0],#1/(12500*100e3),
                    aux_delay=13.34*2/3e8,
                    col_delay = None,
                    T=10e-6,
                    L=12500,
                    B=10e9,
                    min_distance=args.min_distance if args.min_distance is not None else 0,
                    max_distance=args.max_distance if args.max_distance is not None else 30,
                    target_min_distance=args.target_min_distance if args.target_min_distance is not None else 12,
                    target_max_distance=args.target_max_distance if args.target_max_distance is not None else 15,
                    cbf_indices = [[12500//args.downsample, 17500//args.downsample],[18500//args.downsample, 23500//args.downsample]],
                    ref_if_zerocrossings = _turning_point_extension( (18000-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    col_if_zerocrossings = _turning_point_extension( (18000-args.start)//args.downsample, 100//args.downsample, 6250//args.downsample),
                    uprate=10*args.downsample,
                    downsample=args.downsample,
                    )
    else:
        raise ValueError(f"did not recognized {args.data}")

    config["src"] = args.data
    config["slow_fcrm"] = getattr(args, "slow_fcrm", False)
    config["numba_fcrm"] = getattr(args, "numba_fcrm", False)

    config = get_algo_metadata(config, args)

    data = np.column_stack([signal, mzi, t])

    if args.load_if:
        prev_result = np.load(args.load_if)
        ref_if = prev_result["ref_if"]
        if_result = dict(ref_if=ref_if)
        if "col_if" in prev_result.keys():
            col_if = prev_result["col_if"]
            if_result["col_if"] = col_if
            print("successfully loaded auxiliary and collimator IF")
        else:
            print("successfully loaded auxiliary IF")
    else:
        if_result = None
    
    return data, config, if_result


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="triangular", help=("either\n"
                                                         "\"lnoi_edbr_triangular\" or \n"
                                                        "\"0821_laser2_triangular\" or \n"
                                                        "\"0821_laser2_sinusoidal\""))
    p.add_argument("--algo", default="resample", type=str, 
                   help=("\"resampling\" or \"mf\" or \"bp\""))
    p.add_argument("--min-distance", default=None, type=float)
    p.add_argument("--max-distance", default=None, type=float)
    p.add_argument("--target-min-distance", default=None, type=float)
    p.add_argument("--target-max-distance", default=None, type=float)
    p.add_argument("--load-if", default=None, type=str)
    p.add_argument("--inverse", default="ridge", type=str, help="\"ridge\" or \"pinv\" or \"fft\"")
    p.add_argument("--ref-if-est", default="hilbert", type=str, help="\"hilbert\" or \"fcrm\"")
    p.add_argument("--col-if-est", default="none", type=str, help="\"none\" or \"hilbert\" or \"fcrm\"")
    p.add_argument("--subtract-collimator", action="store_true", help="subtract collimator")
    p.add_argument("--slow-fcrm", action="store_true",
                   help="fmcw_aux.estimate_if_fcrm with fast=False (the original\n"
                        "chirplet_ridge.estimate_ifs; same result, slower)")
    p.add_argument("--numba-fcrm", action="store_true",
                   help="fmcw_aux.estimate_if_fcrm with numba=True (compiled kernels;\n"
                        "same result, faster than the default fast=True)")
    p.add_argument("--downsample", default=1, type=int)
    p.add_argument("--workers", type=int, default=None,
                   help="default: all CPUs; 1 = serial in this process")
    p.add_argument("--start", type=int, default=0, help="first row of x to use")
    p.add_argument("--stop", type=int, default=None, help="last row (exclusive)")
    p.add_argument("--max-windows", type=int, default=None,
                   help="only run the first N windows (quick tests)")
    p.add_argument("--on-error", default="raise", choices=["raise", "skip"],
                   help="raise: stop at first failing window; "
                        "skip: continue, failed windows saved as NaN")
    p.add_argument("--out", default=None,
                   help="output .npz (default: results/results_<date>_<time>.npz next to run.py)")
    args = p.parse_args()
    if args.slow_fcrm and args.numba_fcrm:
        p.error("--slow-fcrm and --numba-fcrm exclude each other")

    x, meta, prev = load_data(args)

    if "L" not in meta:
        p.error("set meta['L'] in load_data() or pass --L")
    L = meta["L"]
    win, hop = WIN_BLOCKS * L, HOP_BLOCKS * L

    params = getattr(algorithm, "PARAMS", {})   # optional extra kwargs
    # previous results: window j gets process_segment(..., prev_result={key: prev[key][j]})
    per_window = None if prev is None else dict(prev_result=prev)
    if (meta.get("numba_fcrm") and "fcrm" in (meta.get("ref_if_est"), meta.get("col_if_est"))
            and args.workers != 1):
        # Compile the numba kernels (or load them from their disk cache) here, on
        # the first window, so that the worker processes only load them instead of
        # all compiling at once. Errors are left to the run itself to report.
        try:
            kw = {} if prev is None else dict(prev_result={k: v[0] for k, v in prev.items()})
            algorithm.process_segment(x[args.start:args.start + win].copy(), 0, meta,
                                      **params, **kw)
        except Exception:
            pass
    starts, outputs, errors = run_windows(
        x, win, hop, algorithm.process_segment, meta=meta, params=params,
        per_window=per_window,
        n_workers=args.workers, start=args.start, stop=args.stop,
        max_windows=args.max_windows, on_error=args.on_error)

    out = args.out or time.strftime("results_%Y%m%d_%H%M%S.npz")
    if not os.path.isabs(out):
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        out = os.path.join(HERE, "results", out)
    save_results(out, starts, outputs, errors, meta=meta, L=L, win=win, hop=hop)


if __name__ == "__main__":   # required for multiprocessing on macOS/Windows
    main()
