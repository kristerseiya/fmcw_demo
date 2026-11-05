"""
Rules (so it works in parallel):
  * `process_segment(segment, seg_idx, meta)` must be a top-level function.
  * segment = x[idx : idx + 3*L, :]  (a copy; shape (3L, K)), where
    idx = seg_idx * L (+ --start), i.e. consecutive windows overlap and shift by L
    (window layout is set by WIN_BLOCKS / HOP_BLOCKS in run.py).
  * meta is the dict returned by run.load_data().
  * prev_result is {key: prev[key][seg_idx]} for the dict of previous results
    prev returned by run.load_data() (None if load_data() returned None).
  * It must return a dict whose values are scalars or arrays, e.g.
        return dict(segment_t=segment_t, d=d, s=s)
    Each key is stacked over windows (in order of idx) and saved to .npz.
    Arrays whose length varies between windows are NaN-padded; a value of
    None means "missing" and is saved as NaN.
  * It must not depend on other windows or on shared mutable state.
  * Optional: define PARAMS = dict(...) to pass extra keyword arguments.
"""
import numpy as np
import scipy
from scipy.signal import find_peaks
from scipy.signal import windows, hilbert
from scipy.fft import fft
import algo.sparse
import algo.filter

import fmcw_aux
from filters import SignalModel

# Properties of the data go in `meta` (see run.load_data). Optional tuning
# knobs can go in a PARAMS dict here (passed to process_segment as kwargs).

def estimate_inst_freq(segment, seg_idx, config, return_intermediate=False, prev_result=None):
    """
    segment : np.ndarray, shape (L,)   -- array[seg_idx*L : (seg_idx+1)*L]
    seg_idx : int                      -- which segment this is (0, 1, 2, ...)
    meta    : dict                     -- parameters of the whole vector
                                          (from load_data), e.g. meta["fs"]
    prev_result : dict or None         -- results of a previous run for this
                                          segment: {key: prev[key][seg_idx]}
                                          for the dict prev returned by
                                          load_data; None if prev is None

    Returns whatever you want; outputs are collected in segment order.
    """
    """
    # Example
    fs = meta["fs"]
    L = len(segment)
    t0 = seg_idx * L / fs          # start time of this segment

    # ---- example algorithm: peak frequency of the segment ----
    w = np.hanning(L) if window == "hann" else np.ones(L)
    spec = np.abs(np.fft.rfft(segment * w))
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return dict(t0=t0, peak_freq=freqs[np.argmax(spec)], peak_mag=spec.max())
    """
    del_t = config["del_t"]

    signal_i = segment[:, 0]
    ref_signal_i = segment[:, 1]

    # 1.1 IF estimation
    if config["ref_if_est"] == "fcrm":

        ref_ift, ref_ifq = fmcw_aux.estimate_if_fcrm(ref_signal_i, **config["ref_if_est_params"])
        ref_ift *= del_t
        ref_ifq /= del_t

    elif config["ref_if_est"] == "hilbert":

        ref_ift, ref_ifq = fmcw_aux.estimate_if_hilbert(ref_signal_i, **config["ref_if_est_params"])
        ref_ift *= del_t
        ref_ifq /= del_t

    else:
        raise ValueError(f"could not recognize {config['if_est']} as if estimation type")

    ref_if_smooth_length = config["ref_if_smooth_length"]
    if ref_if_smooth_length > 0:
        ref_ifq = np.pad(ref_ifq, pad_width=(ref_if_smooth_length, ref_if_smooth_length), mode="wrap")
        ref_ifq = np.convolve(ref_ifq, np.ones((2*ref_if_smooth_length+1,))/(2*ref_if_smooth_length+1), mode="valid")

    # 1.2 (collimator) IF estimation
    if config["col_if_est"] != "none":

        if config["col_if_est"] == "fcrm":

            col_ift, col_ifq = fmcw_aux.estimate_if_fcrm(signal_i, **config["col_if_est_params"])
            col_ift *= del_t
            col_ifq /= del_t

        elif config["col_if_est"] == "hilbert":
            col_ift, col_ifq = fmcw_aux.estimate_if_hilbert(signal_i, **config["col_if_est_params"])
            col_ift *= del_t
            col_ifq /= del_t

        col_if_smooth_length = config["col_if_smooth_length"]
        if col_if_smooth_length > 0:
            col_ifq = np.pad(col_ifq, pad_width=(col_if_smooth_length, col_if_smooth_length), mode="wrap")
            col_ifq = np.convolve(col_ifq, np.ones((2*col_if_smooth_length+1,))/(2*col_if_smooth_length+1), mode="valid")

        if len(col_ifq) == len(ref_ifq):
            if np.mean((col_ifq+ref_ifq)**2) < np.mean((col_ifq-ref_ifq)**2):
                col_ifq *= -1
        else:
            ifq_resamp = np.interp(col_ift, ref_ift, ref_ifq)
            if np.mean((col_ifq+ifq_resamp)**2) < np.mean((col_ifq-ifq_resamp)**2):
                col_ifq *= -1

    result = dict()

    result["ref_if"] = np.column_stack([ref_ift, ref_ifq])
    if config["col_if_est"] != "none":
        result["col_if"] = np.column_stack([col_ift, col_ifq])

    return result

def process_segment(segment, seg_idx, config, return_intermediate=False, prev_result=None):
    """
    segment : np.ndarray, shape (L,)   -- array[seg_idx*L : (seg_idx+1)*L]
    seg_idx : int                      -- which segment this is (0, 1, 2, ...)
    meta    : dict                     -- parameters of the whole vector
                                          (from load_data), e.g. meta["fs"]
    prev_result : dict or None         -- results of a previous run for this
                                          segment: {key: prev[key][seg_idx]}
                                          for the dict prev returned by
                                          load_data; None if prev is None

    Returns whatever you want; outputs are collected in segment order.
    """
    """
    # Example
    fs = meta["fs"]
    L = len(segment)
    t0 = seg_idx * L / fs          # start time of this segment

    # ---- example algorithm: peak frequency of the segment ----
    w = np.hanning(L) if window == "hann" else np.ones(L)
    spec = np.abs(np.fft.rfft(segment * w))
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return dict(t0=t0, peak_freq=freqs[np.argmax(spec)], peak_mag=spec.max())
    """
    del_t = config["del_t"]
    T = config["T"]
    L = config["L"]
    B = config["B"]
    aux_delay, col_delay = config["aux_delay"], config["col_delay"]
    target_min_dist, target_max_dist = config["target_min_distance"], config["target_max_distance"]

    signal_i = segment[:, 0]
    ref_signal_i = segment[:, 1]
    ts_i = segment[:, 2]
    ts_i = ts_i - ts_i[0]

    if config["method"] == "resample":

        segment_t = []
        d, s = [], []
        for cbf_idx1, cbf_idx2 in config["cbf_indices"]:

            signal_i_cbf = signal_i[cbf_idx1:cbf_idx2]
            ref_signal_i_cbf = ref_signal_i[cbf_idx1:cbf_idx2]
            ref_cmplx = hilbert(ref_signal_i_cbf)
            v = np.unwrap(np.angle(ref_cmplx))
            v_ideal = np.linspace(v[0], v[-1], len(v))
            interpolater = scipy.interpolate.interp1d(v, np.arange(len(v)), kind="linear")
            new_t = interpolater(v_ideal)
            new_t[new_t>(len(v)-1)] = len(v)-1
            new_t[new_t<0] = 0
            interpolater = scipy.interpolate.interp1d(np.arange(len(v)), signal_i_cbf, kind="linear")
            signal_i_new = interpolater(new_t)
            freq = np.arange(len(signal_i_new)) / len(signal_i_new) / del_t
            spectrum = np.abs(fft(signal_i_new*windows.hann(len(signal_i_new)), norm="ortho"))
            freq = np.arange(len(spectrum)) / len(spectrum) / del_t
            dist_spectrum = freq / (B/T) * 3e8 / 4
            search_region = (dist_spectrum>target_min_dist)*(dist_spectrum<target_max_dist)
            peaks, _ = find_peaks(spectrum[search_region], distance=10)
            peaks = peaks + np.argmax(search_region)
            a = np.argsort(spectrum[peaks])[-10:]
            di = dist_spectrum[peaks][a]
            si = spectrum[peaks][a]
            if len(a) < 10:
                di = np.pad(di, (0,10-len(a)), mode="constant", constant_values=np.nan)
                si = np.pad(si, (0,10-len(a)), mode="constant", constant_values=0)
            segment_t.append(np.mean(ts_i[cbf_idx1:cbf_idx2]))
            d.append(di)
            s.append(si)
        segment_t, d, s, = np.array(segment_t), np.array(d), np.array(s)
        result = dict(segment_t=segment_t, 
                      distances=d, peak_strength=s)
        if return_intermediate:
            result["before_after_resample"] = [ signal_i_cbf, signal_i_new ]

    elif config["method"] in ["if", "mf", "bmf", "bp", "bbp"]:

        uprate = config["uprate"]

        # 1. IF estimation
        if prev_result:
            result = dict()
            if ("ref_if" in prev_result.keys()):
                ref_ift, ref_ifq = prev_result["ref_if"][:,0], prev_result["ref_if"][:, 1]
                ref_if_smooth_length = config["ref_if_smooth_length"]
                if ref_if_smooth_length > 0:
                    ref_ifq = np.pad(ref_ifq, pad_width=(ref_if_smooth_length, ref_if_smooth_length), mode="wrap")
                    ref_ifq = np.convolve(ref_ifq, np.ones((2*ref_if_smooth_length+1,))/(2*ref_if_smooth_length+1), mode="valid")                
                result["ref_if"] = np.column_stack([ref_ift, ref_ifq])
            if (config["col_if_est"] != "none") and ("col_if" in prev_result.keys()):
                col_ift, col_ifq = prev_result["col_if"][:,0], prev_result["col_if"][:, 1]
                col_if_smooth_length = config["col_if_smooth_length"]
                if col_if_smooth_length > 0:
                    col_ifq = np.pad(col_ifq, pad_width=(col_if_smooth_length, col_if_smooth_length), mode="wrap")
                    col_ifq = np.convolve(col_ifq, np.ones((2*col_if_smooth_length+1,))/(2*col_if_smooth_length+1), mode="valid")                
                result["col_if"] = np.column_stack([col_ift, col_ifq])
        else:
            result = estimate_inst_freq(segment, seg_idx, config)
            ref_ift, ref_ifq = result["ref_if"][:,0], result["ref_if"][:, 1]
            if (config["col_if_est"] != "none") and ("col_if" in result.keys()):
                col_ift, col_ifq = result["col_if"][:,0], result["col_if"][:, 1]

        # 2. (trasmitted) IF estimation
        modufuncsolver = config["modfsolver"]
        if config["col_if_est"] == "none":
            a_est = modufuncsolver.estimate((ref_ift, ref_ifq*del_t))
        else:
            a_est = modufuncsolver.estimate((ref_ift, ref_ifq*del_t), (col_ift, col_ifq*del_t))
        a_est /= del_t

        modf_smooth_length = config["modf_smooth_length"]
        if modf_smooth_length > 1:
            a_est = np.pad(a_est, pad_width=(modf_smooth_length, modf_smooth_length), mode="wrap")
            a_est = np.convolve(a_est, np.ones((2*modf_smooth_length+1,))/(2*modf_smooth_length+1), mode="valid")

        result["modf"] = a_est

        if config["method"] == "if":
            return result

        # 3. subtracting collimator
        if config["subtract_collimator"]:
            if config["col_if_est"] == "none":
                raise RuntimeError("collimator if must be estimated in order for subtraction")
            signal_i = fmcw_aux.subtract_signal(ts_i, signal_i, col_ift, col_ifq, 200, period=3*T)

        # 4. matched filtering

        model = SignalModel((ts_i, a_est), model_type="inst_freq", period=3*T)
        y = signal_i*model.get_complex_waveform(ts_i)*(ts_i>=T)*(ts_i<T*2)

        if config["method"] == "mf":

            delays, confidence = algo.filter.matchedfilter(model, ts_i, y, uprate=uprate, **config["algo_params"])
            confidence /= np.amax(confidence)

        elif config["method"] == "bmf":

            delays, confidence = algo.filter.bartlett_matchedfilter(model, ts_i, y, uprate=uprate, **config["algo_params"])
            confidence /= np.amax(confidence)

        elif config["method"] == "bp":

            bp = algo.sparse.basis_pursuit_faster(model, ts_i, y, uprate=uprate, **config["algo_params"])
            confidence = np.abs(bp)
            confidence /= np.amax(confidence)
            delays = np.arange(len(bp))/uprate*del_t

        elif config["method"] == "bbp":

            bbp = algo.sparse.bartlett_basis_pursuit_faster(model, ts_i, y, uprate=uprate, **config["algo_params"])
            confidence = np.mean(np.abs(bbp), axis=0)
            confidence /= np.amax(confidence)
            delays = np.arange(len(confidence))/uprate*del_t
        search_region = (delays>target_min_dist*2/3e8)*(delays<target_max_dist*2/3e8)
        peaks, _ = find_peaks(confidence[search_region], **config["find_peaks_params"])
        peaks = peaks + np.argmax(search_region)
        a = np.argsort(confidence[peaks])[-config["top_k"]:]
        d = (delays*3e8/2)[peaks][a]
        s = confidence[peaks][a]
        if len(a) < config["top_k"]:
            d = np.pad(d, (0,config["top_k"]-len(a)), mode="constant", constant_values=np.nan)
            s = np.pad(s, (0,config["top_k"]-len(a)), mode="constant", constant_values=0)

        segment_t = np.mean(ts_i)
        result["segment_t"]=segment_t
        result["distances"]=d
        result["peak_strength"]=s
        if return_intermediate:
            result["confidence"] = np.column_stack([delays[search_region], confidence[search_region]])
            if config["subtract_collimator"]:
                result["subtracted_signal"] = signal_i

    else:
        raise ValueError(f"did not recongnize {config['method']}")

    return result
