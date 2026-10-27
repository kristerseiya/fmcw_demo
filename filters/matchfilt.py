
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize, minimize_scalar
import scipy.integrate
import scipy.interpolate

from typing import List, Literal, Union

from chirpgen.config import ChirpGeneratorConfig
from chirpgen.modulation import get_modulation


def construct_matchfilter(t:np.ndarray, config:ChirpGeneratorConfig, normalize=False):
    modf = get_modulation(config)
    mf = modf.generate_waveform(t, 0, 0)
    if normalize:
        mf /= np.sqrt(np.abs(mf @ np.conj(mf)))
    return mf

class SignalModel():
    def __init__(self, 
                 model:Union[List,ChirpGeneratorConfig],
                 model_type:Literal["inst_freq", "phase", "config"]="inst_freq",
                 period:float=None):

        self.model_type = model_type
        if model_type == "inst_freq":
            t, ifq = model
            self.t = np.copy(t)
            self.inst_freq = np.copy(ifq)
            self.period = period
        elif model_type == "phase":
            raise NotImplementedError()
        elif model_type == "config":
            self.modf = get_modulation(model)
        else:
            raise ValueError()

    def get_complex_waveform(self, t:np.ndarray):
        if self.model_type == "inst_freq":
            ifq = np.interp(t, self.t, self.inst_freq, period=self.period)
            phase = (2*np.pi)*scipy.integrate.cumulative_simpson(ifq, x=t, dx=t[1]-t[0])
            return np.exp(1j*phase)
        elif self.model_type == "phase":
            raise NotImplementedError()
        elif self.model_type == "config":
            return self.modf.generate_waveform(t, 0, 0)
        else:
            raise RuntimeError()

    def get_real_waveforms(self, t:np.ndarray, n_phase_offsets:int=4):
        phase_offsets = np.linspace(0, 2*np.pi, n_phase_offsets+1)[:-1]
        cmplx_waveform = self.get_complex_waveform(t)
        return np.real(np.exp(1j*phase_offsets)[:, None] * cmplx_waveform)



        
        

    # def get_cmplx_waveform(self):


