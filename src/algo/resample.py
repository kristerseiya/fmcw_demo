
import numpy as np
from scipy.signal import hilbert

def resampling_by_linerized_phase(reference, x):
    rh = hilbert(reference)
    v = np.unwrap(np.angle(rh))
    v_linear = np.linspace(v[0], v[-1], len(v))
    t = np.arange(len(v))
    tnew = np.interp(v_linear, v, t)
    xnew = np.interp(tnew, t, x)
    return xnew

def resampling_by_zero_crossings(reference, x, interpolate=True):
    reference = reference - np.mean(reference)
    zero_crossings = np.where(np.diff(np.sign(reference)))[0]
    if interpolate:
        y1 = reference[zero_crossings+1]
        y0 = reference[zero_crossings]
        tnew = zero_crossings-y0/(y1-y0)
        t = np.arange(len(x))
        xnew = np.interp(tnew, t, x)
    else:
        xnew = x[zero_crossings]
    return xnew