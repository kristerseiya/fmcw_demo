
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike

import utils

class ChirpGeneratorConfig():
    def __init__(self, Tchirp=10e-6, Trep=100e-6,
                 sample_rate=1e8, bandwidth=0.1e8, lambd_c=1500e-9, 
                 modulation="triangle", 
                 modulation_data=None,
                 generation_type="pulse",
                 dechirp=False,
                 phase_noise_stdv=1, additive_noise_stdv=1, distance_fall_off=True,
                 phase_noise_type="white",
                 acf_fun=None,
                 psd_fun=None,
                 complex_valued=True,
                 phase_noise_gen_delta_t=None,
                 transmitted_power=1e-3, detector_effectivity=1, detector_effective_area=1e-6):
        
        if generation_type == "pulse" and Tchirp > Trep:
            raise ValueError("Trep should be larger than Tchirp for pulse generation")
        
        self.sample_rate = sample_rate
        self.Tchirp = Tchirp
        self.Trep = Trep
        self.bandwidth = bandwidth
        self.lambd_c = lambd_c
        self.modulation = modulation
        self.modulation_data = modulation_data
        self.max_d = 2*self.Tchirp * 3e8 / 2
        self.max_v = self.sample_rate * self.lambd_c / 4
        self.transmitted_power = transmitted_power
        self.detector_effectivity = detector_effectivity
        self.detector_effective_area = detector_effective_area
        self.additive_noise_stdv = additive_noise_stdv
        self.phase_noise_stdv = phase_noise_stdv
        self.complex_valued = complex_valued
        self.dechirp = dechirp
        self.generation_type = generation_type
        self.phase_noise_type = phase_noise_type
        self.distance_fall_off = distance_fall_off
        self.acf_fun = acf_fun
        self.psd_fun = psd_fun
        self.phase_noise_gen_delta_t = phase_noise_gen_delta_t

    def get_sample_rate(self):
        return self.sample_rate

    def get_chirp_length(self):
        return self.Tchirp

    def get_bandwidth(self):
        return self.bandwidth
    
    def get_carrier_wavelength(self):
        return self.lambd_c
    
    def get_modulation_type(self):
        return self.modulation
    
    def get_max_d(self):
        return self.max_d
    
    def get_max_v(self):
        return self.max_v
    
    def is_complex_valued(self):
        return self.complex_valued

