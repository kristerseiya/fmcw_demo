
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, ifft
from typing import Sequence
import copy
from scipy.signal import convolve
from scipy.interpolate import interp1d

import utils
from .config import ChirpGeneratorConfig

class Modulation():
    def __init__(self):
        pass
    def generate_phase_x(self, distance: float, velocity: float):
        raise NotImplementedError()
    
    def generate_freq_x(self):
        raise NotImplementedError()
    
    # def compute_jacobian_x(self):
    #     raise NotImplementedError()
    
    def generate_phase(self, distance: float, velocity: float):
        return self.generate_phase_x(0, 0) - self.generate_phase_x(distance, velocity)
    
    def generate_freq(self, distance: float, velocity: float):
        return self.generate_freq_x(0, 0) - self.generate_freq_x(distance, velocity)
    
    # def compute_jacobian(self, distance: float, velocity: float):
    #     return self.compute_jacobian_x(0, 0) - self.compute_jacobian_x(distance, velocity)

class LinearModulation(Modulation):
    def __init__(self, meas_prop: ChirpGeneratorConfig):
        if meas_prop.get_modulation_type() != 'triangle':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()

    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
        sample_rate = self.meas_prop.get_sample_rate()
        doppler_shift = velocity * 2 / self.lambd_c
        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        t_wrap = utils.wrap(t-delay, 0, self.Tchirp)
        triangle_if_rx = gamma*t_wrap - doppler_shift

        if normalize_freq:
            triangle_if_rx  /= sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for triangular modulation generate_freq_x is not supported')

        return triangle_if_rx
    
    def generate_phase(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0,
                         compute_jacobian: bool=False):

        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        ts_w, mirror_deriv = utils.mirror_deriv(t-delay-v*t, 0, self.Tchirp)
        phase = 0.5*gamma*(ts_w**2)
        period_num = np.floor((t-delay-v*t)/self.Tchirp).astype(int)
        phase = phase* (-np.ones(t.shape))**(period_num)
        phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(self.Tchirp**2)
        phase_offset2 = period_num * 0.5*gamma*(self.Tchirp**2)
        phase = phase + phase_offset1 + phase_offset2 - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = gamma*ts_w*mirror_deriv*(-2/3e8)
                jacobian[:,:,1] = gamma*ts_w*mirror_deriv*(-2*t/3e8)
                jacobian[:,:,1] += -f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = gamma*ts_w*mirror_deriv*(-2/3e8)
                jacobian[:,1] = gamma*ts_w*mirror_deriv*(-2*t/3e8)
                jacobian[:,1] += -f0*(2*t/3e8)
            jacobian *= 2*np.pi
            return phase, jacobian
        
        return phase
    
class SinusoidalModulation(Modulation):
    def __init__(self, config: ChirpGeneratorConfig):
        if config.get_modulation_type() != 'sinusoidal':
            raise ValueError('measurement property is not consistent with the class')
        self.config = copy.deepcopy(config)
        self.sample_rate = config.get_sample_rate()
        self.Tchirp = config.get_chirp_length()
        self.Trep = config.Trep
        self.bandwidth = config.get_bandwidth()
        self.lambd_c = config.get_carrier_wavelength()

    def generate_freq(self, t:np.ndarray, 
                      distance:float|np.ndarray, 
                      velocity:float|np.ndarray=0, 
                      normalize_freq:bool=True):
        
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        doppler_shift = - velocity * 2 / self.lambd_c
        delay = (distance+velocity*t) * 2 / 3e8
        Trep = self.config.Trep
        Tchirp = self.config.Tchirp

        generation_type = self.config.generation_type

        if generation_type == "pulse":

            inst_freq = np.ones((len(delay), len(t)), dtype=float) * np.nan
            N_start = int(np.floor(t[0] / Trep))
            N_end = int(np.ceil(t[-1] / Trep))
            for n in range(N_start, N_end):
                t_tmp = t-n*Trep-delay
                inst_freq_tmp = - (1-velocity*2/3e8) * 0.5 * self.bandwidth * np.cos(np.pi/Tchirp*(t_tmp)) + doppler_shift + 0.5 * self.bandwidth
                pulse_idx = (t_tmp>=0)*(t_tmp<Tchirp)
                inst_freq[pulse_idx] = inst_freq_tmp[pulse_idx]

        elif generation_type == "continuous":
            t_tmp = t-delay
            inst_freq = - (1-velocity*2/3e8) * 0.5 * self.bandwidth * np.cos(np.pi/Tchirp*t_tmp) + doppler_shift + 0.5 * self.bandwidth

        if normalize_freq:
            inst_freq /= self.sample_rate

        if len(distance) == 1:
            inst_freq = inst_freq[0]
        
        return inst_freq
    
    def generate_phase(self, t:np.ndarray, 
                       distance:float|np.ndarray=0, 
                       velocity: float|np.ndarray=0):
        
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        delay = (distance+velocity*t) * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if self.lambd_c == 0:
            f0 = 0
        else:
            f0 = 3e8 / self.lambd_c

        generation_type = self.config.generation_type

        if generation_type == "pulse":

            Trep = self.config.Trep
            Tchirp = self.config.Tchirp
            phase = np.ones((len(delay), len(t)), dtype=float) * np.nan
            N_start = int(np.floor(t[0] / Trep))
            N_end = int(np.ceil(t[-1] / Trep))
            for n in range(N_start, N_end):
                t_tmp = t-n*Trep-delay
                phase_tmp = -1/np.pi/2*self.Tchirp*self.bandwidth*np.sin(np.pi*t_tmp/self.Tchirp) 
                phase_tmp += (f0 + self.bandwidth/2)*t_tmp
                phase_tmp *= 2*np.pi
                pulse_idx = (t_tmp>=0)*(t_tmp<Tchirp)
                phase[pulse_idx] = phase_tmp[pulse_idx]

        elif generation_type == "continuous":
            t_tmp = t-delay
            phase = -1/np.pi/2*self.Tchirp*self.bandwidth*np.sin(np.pi*t_tmp/self.Tchirp) 
            phase += (f0 + self.bandwidth/2)*t_tmp
            phase *= 2*np.pi

        if len(distance) == 1:
            phase = phase[0]

        return phase
    
    def generate_waveform(self, t:np.ndarray, 
                       distance:float|np.ndarray=0, 
                       velocity: float|np.ndarray=0):

        phase = self.generate_phase(t, distance, velocity)

        waveform = np.exp(1j*phase)
        waveform[np.isnan(waveform)] = 0

        return waveform

class TriangularModulation(Modulation):
    def __init__(self, config: ChirpGeneratorConfig):
        if config.get_modulation_type() != 'triangle':
            raise ValueError('measurement property is not consistent with the class')
        self.config = copy.deepcopy(config)
        self.sample_rate = config.get_sample_rate()
        self.Tchirp = config.get_chirp_length()
        self.bandwidth = config.get_bandwidth()
        self.lambd_c = config.get_carrier_wavelength()

    def generate_freq(self, t: np.ndarray, 
                      distance: float|np.ndarray, 
                      velocity: float|np.ndarray, 
                      normalize_freq: bool=True):
    
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        delay = (distance+velocity*t) * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        doppler_shift = - velocity * 2 / self.lambd_c

        generation_type = self.config.generation_type

        if generation_type == "pulse":

            Trep = self.config.Trep
            Tchirp = self.config.Tchirp
            inst_freq = np.ones((len(distance), len(t)), dtype=float) * np.nan
            N_start = int(np.floor(t[0] / Trep))
            N_end = int(np.ceil(t[-1] / Trep))
            for n in range(N_start, N_end):
                t_tmp = t-n*Trep-delay
                t_mirrored = utils.mirror(t_tmp, 0, self.Tchirp)
                inst_freq_tmp = (1-velocity*2/3e8) * gamma*t_mirrored + doppler_shift
                t_mirrored = utils.mirror(t_tmp, 0, self.Tchirp)
                pulse_idx = (t_tmp>=0)*(t_tmp<Tchirp)
                inst_freq[pulse_idx] = inst_freq_tmp[pulse_idx]

        elif generation_type == "continuous":
            t_tmp = t - delay
            t_mirrored = utils.mirror(t_tmp, 0, self.Tchirp)
            inst_freq = (1-velocity*2/3e8) * gamma*t_mirrored + doppler_shift

        if normalize_freq:
            inst_freq /= self.sample_rate

        if len(distance) == 1:
            inst_freq = inst_freq[0]
        
        return inst_freq
    
    def generate_phase(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0,
                         compute_jacobian: bool=False):
    
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        # delay = distance * 2 / 3e8
        delay = (distance+velocity*t) * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if self.lambd_c == 0:
            f0 = 0
        else:
            f0 = 3e8 / self.lambd_c

        generation_type = self.config.generation_type

        if generation_type == "pulse":

            Trep = self.config.Trep
            Tchirp = self.config.Tchirp
            phase = np.ones((len(delay), len(t)), dtype=float) * np.nan
            N_start = int(np.floor(t[0] / Trep))
            N_end = int(np.ceil(t[-1] / Trep))
            for n in range(N_start, N_end):
                t_tmp = t-n*Trep-delay
                ts_w = utils.mirror(t_tmp, 0, self.Tchirp)
                phase_tmp = 0.5*gamma*(ts_w**2)
                period_num = np.floor(t_tmp/self.Tchirp).astype(int)
                phase_tmp = phase_tmp* (-np.ones(t.shape))**(period_num)
                phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(self.Tchirp**2)
                phase_offset2 = period_num * 0.5*gamma*(self.Tchirp**2)
                phase_tmp = phase_tmp + phase_offset1 + phase_offset2 + f0*t_tmp
                phase_tmp *= 2*np.pi

                pulse_idx = (t_tmp>=0)*(t_tmp<Tchirp)
                phase[pulse_idx] = phase_tmp[pulse_idx]

        elif generation_type == "continuous":
            t_tmp = t-delay
            ts_w = utils.mirror(t_tmp, 0, self.Tchirp)
            phase = 0.5*gamma*(ts_w**2)
            period_num = np.floor(t_tmp/self.Tchirp).astype(int)
            phase = phase* (-np.ones(t.shape))**(period_num)
            phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(self.Tchirp**2)
            phase_offset2 = period_num * 0.5*gamma*(self.Tchirp**2)
            phase = phase + phase_offset1 + phase_offset2 + f0*t_tmp
            phase *= 2*np.pi

        if len(distance) == 1:
            phase = phase[0]

        return phase
    
    def generate_waveform(self, t:np.ndarray, 
                       distance:float|np.ndarray=0, 
                       velocity: float|np.ndarray=0):

        phase = self.generate_phase(t, distance, velocity)

        waveform = np.exp(1j*phase)
        waveform[np.isnan(waveform)] = 0

        return waveform


class CustomModulation(Modulation):
    def __init__(self, config: ChirpGeneratorConfig):
        if config.get_modulation_type() != 'custom':
            raise ValueError('measurement property is not consistent with the class')
        ref_t = np.linspace(-1, 2, len(config.modulation_data))
        ifq = config.modulation_data
        phase = np.cumsum(config.modulation_data)/len(config.modulation_data)*3
        self.ifq_fn = interp1d(ref_t, ifq, "linear")
        self.phase_fn = interp1d(ref_t, phase, "linear")
        # self.offset1 = config.modulation_data[0]
        # self.offset2 = config.modulation_data[-1]
        self.config = config
        self.sample_rate = config.get_sample_rate()
        self.Tchirp = config.get_chirp_length()
        self.bandwidth = config.get_bandwidth()
        self.lambd_c = config.get_carrier_wavelength()

    def generate_freq(self, t: np.ndarray, 
                      distance: float|np.ndarray, 
                      velocity: float|np.ndarray, 
                      normalize_freq: bool=True):
        
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        # delay = distance * 2 / 3e8
        delay = (distance+velocity*t) * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if self.lambd_c == 0:
            f0 = 0
        else:
            f0 = 3e8 / self.lambd_c
        Trep = self.config.Trep

        generation_type = self.config.generation_type

        if generation_type == "pulse":
        
            raise NotImplementedError()

        elif generation_type == "continuous":
            t_tmp = t-delay
            ifq = self.ifq_fn(t_tmp/Trep)*self.bandwidth + f0*velocity*2/3e8

        if len(distance) == 1:
            ifq = ifq[0]

        return ifq
    
    def generate_phase(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0,
                         compute_jacobian: bool=False):
    
        if isinstance(distance, np.ndarray) and np.isscalar(velocity):
            velocity = velocity * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity):
            distance = np.array([distance])
            velocity = np.array([velocity])

        distance = np.expand_dims(distance, 1)
        velocity = np.expand_dims(velocity, 1)

        # delay = distance * 2 / 3e8
        delay = (distance+velocity*t) * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if self.lambd_c == 0:
            f0 = 0
        else:
            f0 = 3e8 / self.lambd_c
        Trep = self.config.Trep

        generation_type = self.config.generation_type

        if generation_type == "pulse":

            raise NotImplementedError()

        elif generation_type == "continuous":
            t_tmp = t-delay
            phase = self.phase_fn(t_tmp/Trep)*self.bandwidth*Trep + f0*t_tmp
            phase *= 2*np.pi

        if len(distance) == 1:
            phase = phase[0]

        return phase
    
    def generate_waveform(self, t:np.ndarray, 
                       distance:float|np.ndarray=0, 
                       velocity: float|np.ndarray=0):

        phase = self.generate_phase(t, distance, velocity)

        waveform = np.exp(1j*phase)
        waveform[np.isnan(waveform)] = 0

        return waveform


def _stairfunc(x,c,e,x0,d,a,compute_deriv=False):
    # c = 30
    # e = 0.5
    # x0 = 1/10
    # d = ( (x0-0.5)+e-1/2*x0 ) / ( np.sin(c*(x0-0.5)) - 0.5*x0*c*np.cos(c*(x0-0.5)) )
    # a = (1- c*d*np.cos(c*(x0-0.5))) / (2*x0)
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    # y[(x>=(1-x0))*(x<(1+x0))] = -a*(x[(x>=(1-x0))*(x<(1+x0))]+1)**2 + 1
    # y[(x>=(x0-1))*(x<(-x0))] = (x[(x>=(x0-1))*(x<(-x0))]+1.5)-d*np.sin(c*(x[(x>=(x0-1))*(x<(-x0))]+1.5)) + e
    xint1 = (x_w>=-x0)*(x_w<x0)
    xint2 = (x_w>=x0)*(x_w<(1-x0))
    xint3 = (x_w>=(1-x0))*(x_w<(1+x0))
    y[xint1] = a*(x_w[xint1])**2
    y[xint2] = (x_w[xint2]-0.5)-d*np.sin(c*(x_w[xint2]-0.5)) + e
    y[xint3] = -a*(x_w[xint3]-1)**2 + 1
    y = y - 0.5
    # y[(x>=(1+x0))*(x<(2-x0))] = -(x[(x>=(1+x0))*(x<(2-x0))]-1.5)+d*np.sin(c*(x[(x>=(1+x0))*(x<(2-x0))]-1.5)) + e
    # y[(x>=(2-x0))] = -(x[(x>=(2-x0))]-1.5)+d*np.sin(c*(x[(x>=(2-x0))]-1.5)) + e
    if compute_deriv:
        z = np.zeros_like(x)
        z[xint1] = 2*a*(x_w[xint1])
        z[xint2] = 1-c*d*np.cos(c*(x_w[xint2]-0.5))
        z[xint3] = -2*a*(x_w[xint3]-1)
        period_num = np.floor(x).astype(int)
        z = z * (-np.ones(z.shape))**(period_num)
        return y, z
    return y

def _stairfunc_integral(x,c,e,x0,d,a):
    
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x)
    # y[(x>=(1-x0))*(x<(1+x0))] = -a*(x[(x>=(1-x0))*(x<(1+x0))]+1)**2 + 1
    # y[(x>=(x0-1))*(x<(-x0))] = (x[(x>=(x0-1))*(x<(-x0))]+1.5)-d*np.sin(c*(x[(x>=(x0-1))*(x<(-x0))]+1.5)) + e
    xint1 = (x_w>=-x0)*(x_w<x0)
    xint2 = (x_w>=x0)*(x_w<(1-x0))
    xint3 = (x_w>=(1-x0))*(x_w<(1+x0))
    y[xint1] = 1/3*a*(x_w[xint1])**3
    y[xint2] = 1/2*(x_w[xint2]-0.5)**2+d/c*np.cos(c*(x_w[xint2]-0.5)) + e*x_w[xint2]
    y[xint2] = y[xint2] - 1/2*(x0-0.5)**2 - d/c*np.cos(c*(x0-0.5)) - e*x0 + 1/3*a*x0**3
    y[xint3] = -1/3*a*(x_w[xint3]-1)**3 + x_w[xint3]
    y[xint3] =  y[xint3] + 1/3*a*(-x0)**3 - (1-x0) + 1/3*a*x0**3 + e*(1-x0) - e*x0 
    
    T = 1/3*a*(-x0)**3 - (1-x0) + 1/3*a*x0**3 + e*(1-x0) - e*x0 + 1
    period_num = np.floor(x).astype(int)
    y = y * (-np.ones(x.shape))**(period_num)
    offset1 = ( (period_num % 2) == 1 ) * T
    offset2 = period_num * T
    y = y + offset1 + offset2
    y = y - 0.5*x

    return y

class SmoothStairsModulation(Modulation):
    def __init__(self, meas_prop: ChirpGeneratorConfig):
        if meas_prop.get_modulation_type() != 'smoothstairs':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()

        self.stair_func_params = {}
        c = 30
        e = 0.5
        x0 = 1/10
        d = ( (x0-0.5)+e-1/2*x0 ) / ( np.sin(c*(x0-0.5)) - 0.5*x0*c*np.cos(c*(x0-0.5)) )
        a = (1- c*d*np.cos(c*(x0-0.5))) / (2*x0)
        self.stair_func_params['c'] = c
        self.stair_func_params['e'] = e
        self.stair_func_params['x0'] = x0
        self.stair_func_params['d'] = d
        self.stair_func_params['a'] = a

    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
        d = distance
        v = velocity

        if isinstance(d , Sequence):
            d = np.array(d)
        if isinstance(v, Sequence):
            v = np.array(v)
        if isinstance(d, np.ndarray):
            d = np.expand_dims(d, 1)
        if isinstance(v, np.ndarray):
            v = np.expand_dims(v, 1)

        doppler_shift = v * 2 / self.lambd_c
        delay = d * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if_rx = self.bandwidth * _stairfunc((t-delay)/self.Tchirp, **self.stair_func_params, compute_deriv=False) - doppler_shift
        if_rx += 0.5 * self.bandwidth

        if normalize_freq:
            if_rx /= self.sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for sinusoidal modulation generate_freq_x is not supported')

        return if_rx
    
    def generate_phase(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                         compute_jacobian: bool=False, compute_hessian: bool=False):
        
        if compute_hessian:
            raise NotImplementedError()
    
        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        phase = self.bandwidth*self.Tchirp * _stairfunc_integral((t-delay-v*t)/self.Tchirp, **self.stair_func_params)
        phase += self.bandwidth/2*(t-delay-v*t) - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2/3e8)
                jacobian[:,:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,:,1] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2*t/3e8) 
                jacobian[:,:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2/3e8)
                jacobian[:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,1] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2*t/3e8) 
                jacobian[:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            jacobian *= 2*np.pi

            return phase, jacobian

        return phase


def _hardsteep(x,interval,compute_deriv=False):
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    xint1 = (x_w<0.5-interval/2)
    xint2 = (x_w>=0.5-interval/2)*(x_w<=0.5+interval/2)
    xint3 = (x_w>0.5+interval/2)
    y[xint1] = -0.5
    y[xint2] = (x_w[xint2]-0.5)/interval
    y[xint3] = 0.5

    if compute_deriv:
        z = np.zeros_like(x)
        z[xint1] = 0
        z[xint2] = 1/interval
        z[xint3] = 0
        period_num = np.floor(x).astype(int)
        z = z * (-np.ones(z.shape))**(period_num)
        return y, z
    return y

def _hardsteep_integral(x,interval):
    
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    xint1 = (x_w<0.5-interval/2)
    xint2 = (x_w>=0.5-interval/2)*(x_w<=0.5+interval/2)
    xint3 = (x_w>0.5+interval/2)

    y[xint2] = 1/2/interval*(x_w[xint2]-0.5)**2 + 0.5*interval/4
    y[xint3] = interval/4 + (x_w[xint3] - 0.5 - interval/2)
    
    T = interval/4 + (1-0.5-interval/2)
    period_num = np.floor(x).astype(int)
    y = y * (-np.ones(x.shape))**(period_num)
    offset1 = ( (period_num % 2) == 1 ) * T
    offset2 = period_num * T
    y = y + offset1 + offset2
    y = y - 0.5*x

    return y

class HardSteepModulation(Modulation):
    def __init__(self, meas_prop: ChirpGeneratorConfig):
        if meas_prop.get_modulation_type() != 'hardsteep':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()

        self.func_params = {}
        self.func_params['interval'] = 200/self.sample_rate/self.Tchirp
        # self.func_params['interval'] = 1

    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
        d = distance
        v = velocity

        if isinstance(d , Sequence):
            d = np.array(d)
        if isinstance(v, Sequence):
            v = np.array(v)
        if isinstance(d, np.ndarray):
            d = np.expand_dims(d, 1)
        if isinstance(v, np.ndarray):
            v = np.expand_dims(v, 1)

        doppler_shift = v * 2 / self.lambd_c
        delay = d * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        if_rx = self.bandwidth * _hardsteep((t-delay)/self.Tchirp, **self.func_params, compute_deriv=False) - doppler_shift
        if_rx += 0.5 * self.bandwidth
        
        if normalize_freq:
            if_rx /= self.sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for sinusoidal modulation generate_freq_x is not supported')

        return if_rx
    
    def generate_phase(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                         compute_jacobian: bool=False, compute_hessian: bool=False):
        
        if compute_hessian:
            raise NotImplementedError()
    
        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        phase = self.bandwidth*self.Tchirp * _hardsteep_integral((t-delay-v*t)/self.Tchirp, **self.func_params)
        phase += self.bandwidth/2*(t-delay-v*t) - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2/3e8)
                jacobian[:,:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,:,1] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2*t/3e8) 
                jacobian[:,:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2/3e8)
                jacobian[:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,1] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2*t/3e8) 
                jacobian[:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            jacobian *= 2*np.pi

            return phase, jacobian

        return phase


    
def get_modulation(config: ChirpGeneratorConfig):
    modulation_type = config.modulation
    if modulation_type == 'triangle':
        return TriangularModulation(config)
    elif modulation_type == 'sinusoidal':
        return SinusoidalModulation(config)
    elif modulation_type == 'smoothstairs':
        return SmoothStairsModulation(config)
    elif modulation_type == 'hardsteep':
        return HardSteepModulation(config)
    elif modulation_type == "custom":
        return CustomModulation(config)
    elif modulation_type == 'mixed':
        raise NotImplementedError()
        # return get_trisin_aliased_mixed_if
    elif modulation_type == 'polynomial':
        raise NotImplementedError()
        # return get_polynomial_aliased_mixed_if
    else:
        raise ValueError('modulation type \"{:s}\" not found'.format(config))
