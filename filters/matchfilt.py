
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize, minimize_scalar

from chirpgen.config import ChirpGeneratorConfig
from chirpgen.modulation import get_modulation

def construct_matchfilter(t:np.ndarray, config:ChirpGeneratorConfig, normalize=False):
    modf = get_modulation(config)
    mf = modf.generate_waveform(t, 0, 0)
    if normalize:
        mf /= np.sqrt(np.abs(mf @ np.conj(mf)))
    return mf

