
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike

from . import modulation as modf
import utils
from .config import ChirpGeneratorConfig

# def gen_phase_noise(ts: ArrayLike, 
#                     distance: float|ArrayLike, 
#                     velocity: float|ArrayLike, 
#                     linewidth: float,) -> np.ndarray[float]:

#     if np.isscalar(distance) and np.isscalar(velocity):
#         distance = np.array([distance])
#         velocity = np.array([velocity])
#     else:
#         if len(distance) != len(velocity):
#             raise ValueError("Distance and velocity must have same length")
#         distance = np.array(distance)
#         velocity = np.array(velocity)

#     n_echo = len(distance)
#     N_t = len(ts)

#     tt = ts -  2*(np.reshape(distance,(-1,1))+np.reshape(velocity,(-1,1))*ts)/3e8

#     ts_all = np.concatenate([ts, *[tt[i] for i in range(tt.shape[0])]])
#     sort_idx = np.argsort(ts_all)
#     ts_all = ts_all[sort_idx]

#     phase_noise = np.zeros_like(ts_all)
#     phase_noise[0] = np.random.rand() * 2*np.pi
    
#     for i in range(1, 2*N_t):
#         delay = ts_all[i] - ts_all[i-1]
#         phase_noise[i] = phase_noise[i-1] + np.random.normal(0, scale=np.sqrt(np.abs(delay)*linewidth*2*np.pi))

#     phase_noise_mix = np.zeros((n_echo, N_t))
#     tx_idx = (sort_idx >= 0) * (sort_idx < N_t)
#     phase_noise_tx = phase_noise[tx_idx]
#     for n in range(n_echo):
#         rx_idx = (sort_idx >= (n+1)*N_t) * (sort_idx < (n+2)*N_t)
#         phase_noise_rx = phase_noise[rx_idx]
#         phase_noise_mix[n] = phase_noise_tx - phase_noise_rx

#     if n_echo == 1:
#         phase_noise_mix = phase_noise_mix[0]

#     return phase_noise_mix

#     # num_t = len(ts)
#     # tt = ts - tau
#     # ts_all = np.zeros((num_t*(n_echo+1)))
#     # idx_ts = np.zeros((num_t,), dtype=int)
#     # idx_tt = np.zeros((n_echo, num_t), dtype=int)
#     # i = 0 # idx for ts
#     # j = 0 # idx for tt
#     # k = 0 # idx for ts_all
#     # while j < num_t and tt[j] < ts[i]:
#     #     ts_all[k] = tt[j]
#     #     idx_tt[j] = k
#     #     j = j + 1
#     #     k = k + 1
#     # while j < num_t:
#     #     ts_all[k] = ts[i]
#     #     ts_all[k+1] = tt[j]
#     #     idx_ts[i] = k
#     #     idx_tt[j] = k + 1
#     #     i = i + 1
#     #     j = j + 1
#     #     k = k + 2
#     # while i < num_t:
#     #     ts_all[k] = ts[i]
#     #     idx_ts[i] = k
#     #     i = i + 1
#     #     k = k + 1

#     # phi_n = np.zeros_like(ts_all)
#     # phi_n[0] = np.random.rand() * 2*np.pi
#     # for i in range(1, 2*num_t):
#     #     delay = ts_all[i] - ts_all[i-1]
#     #     phi_n[i] = phi_n[i-1] + np.random.normal(0, scale=np.sqrt(np.abs(delay)*linewidth*2*np.pi))

#     # del_phi_n = phi_n[idx_ts] - phi_n[idx_tt]

#     # if return_all:
#     #     return del_phi_n, phi_n[idx_ts], phi_n[idx_tt]


#     # return del_phi_n

def compute_lorentz(freq, f0, del_freq, f_sample):
    return (del_freq/f_sample**2) * 2*np.pi / (2*np.pi**2 * ( (freq-f0)**2 + (del_freq/f_sample)**2 ) )

def gaussian_weighted_score(tf_plot, sigma, mean, freq=None):
    n_row, n_col = tf_plot.shape
    score_col = np.zeros((n_col))
    if freq is None:
        x = np.linspace(0, 1, n_row)
    else:
        x = freq
    for i_col in range(n_col):
        weights = np.exp(- (x-mean[i_col])**2 / sigma**2 / 2 )
        weights = weights / np.sum(weights)
        score_col[i_col] = np.sum(tf_plot[:,i_col] * weights)
    score = np.sum(score_col)
    return score

# class FMCWMeasurementProperties():
#     def __init__(self, sample_rate, Tchirp, bandwidth, linewidth, lambd_c, modulation, 
#                  mix_coef=0.99, poly_coefs_up=None, poly_coefs_down=None, assume_zero_velocity=False,
#                  include_shot_noise=True, transmitted_power=1e-3, reflectance=0.1, detector_effectivity=1, detector_effective_area=1e-6, complex_available=True):
#         self.sample_rate = sample_rate
#         self.Tchirp = Tchirp
#         self.bandwidth = bandwidth
#         self.linewidth = linewidth
#         self.lambd_c = lambd_c
#         self.modulation = modulation
#         self.mix_coef = mix_coef
#         self.poly_coefs_up = poly_coefs_up
#         self.poly_coefs_down = poly_coefs_down
#         self.max_d = 2*self.Tchirp * 3e8 / 2
#         self.max_v = self.sample_rate * self.lambd_c / 4
#         self.transmitted_power = transmitted_power
#         self.assume_zero_velocity = assume_zero_velocity
#         self.reflectance = reflectance
#         self.detector_effectivity = detector_effectivity
#         self.detector_effective_area = detector_effective_area
#         self.include_shot_noise = include_shot_noise
#         self.complex_available = complex_available
#         self.boundary_constraint = False
#         self.d_range = (0, self.max_d)
#         self.v_range = (-self.max_v, self.max_v)
#         # by default lower limit is not included in the interval but upper limit is

#     def compute_expected_snr(self, distance):
#         eta = self.detector_effectivity
#         R = self.reflectance
#         A = self.detector_effective_area
#         tx_power = self.transmitted_power
#         lo_power = self.transmitted_power
#         rx_power = tx_power * A / (np.pi*distance**2) * R
#         q = 1.6e-19
#         return 10*np.log10((2 * eta**2 *lo_power *rx_power)/(1*q*eta*(lo_power+rx_power)))

#     def compute_expected_total_power(self, distance):
#         eta = self.detector_effectivity
#         R = self.reflectance
#         A = self.detector_effective_area
#         tx_power = self.transmitted_power
#         lo_power = self.transmitted_power
#         rx_power = tx_power * A / (np.pi*distance**2) * R
#         q = 1.6e-19
#         return (2* eta**2 *lo_power *rx_power)+(1*q*eta*(lo_power+rx_power))
    
#     def compute_expected_signal_power(self, distance):
#         eta = self.detector_effectivity
#         R = self.reflectance
#         A = self.detector_effective_area
#         tx_power = self.transmitted_power
#         lo_power = self.transmitted_power
#         rx_power = tx_power * A / (np.pi*distance**2) * R
#         q = 1.6e-19
#         if self.is_complex_available():
#             expected_signal_power = (2 * eta**2 *lo_power *rx_power)
#         else:
#             expected_signal_power = (2* eta**2 *lo_power *rx_power)
#         return expected_signal_power

#     def compute_expected_shotnoise_power(self, distance):
#         eta = self.detector_effectivity
#         R = self.reflectance
#         A = self.detector_effective_area
#         tx_power = self.transmitted_power
#         lo_power = self.transmitted_power
#         rx_power = tx_power * A / (np.pi*distance**2) * R
#         q = 1.6e-19
#         if self.is_complex_available():
#             expected_shotnoise_power = (4*q*eta*(lo_power+rx_power))
#         else:
#             expected_shotnoise_power = (2*q*eta*(lo_power+rx_power))
#         return expected_shotnoise_power

#     def set_boundary_constraint(self, d_range: Sequence[float]|None=None, v_range: Sequence[float]|None=None):
#         if d_range is not None:
#             if len(d_range) != 2:
#                 raise ValueError('d_range must be length of 2')
#             if d_range[0] < 0:
#                 raise ValueError('lower limit should not be less than 0')
#             if d_range[1] > self.max_d:
#                 raise ValueError('upper limit should not be more than {:.2f}'.format(self.max_d))
#             self.d_range = d_range
            
#         if v_range is not None:
#             if len(v_range) != 2:
#                 raise ValueError('v_range must be length of 2')
#             if v_range[0] < -self.max_v:
#                 raise ValueError('lower limit should not be less than 0')
#             if v_range[1] > self.max_v:
#                 raise ValueError('upper limit should not be more than {:.2f}'.format(self.max_d))
#             self.v_range = v_range
#         self.boundary_constraint = True

#     def remove_boundary_constraint(self):
#         self.d_range = (0, self.max_d)
#         self.v_range = (-self.max_v, self.max)
#         self.boundary_constraint = False
            
#     def set_max_range(self, percentage):
#         self.max_d = (2*self.Tchirp * percentage - 1/self.sample_rate) * 3e8 / 2
#         self.full_range = False

#     def get_range(self):
#         return self.d_range, self.v_range

#     def get_sample_rate(self):
#         return self.sample_rate

#     def get_chirp_length(self):
#         return self.Tchirp

#     def get_bandwidth(self):
#         return self.bandwidth
    
#     def get_linewidth(self):
#         return self.linewidth
    
#     def get_carrier_wavelength(self):
#         return self.lambd_c
    
#     def get_modulation_type(self):
#         return self.modulation
    
#     def get_mix_coef(self):
#         return self.mix_coef
    
#     def get_poly_coefs(self):
#         return self.poly_coefs_up, self.poly_coefs_down
    
#     def get_max_d(self):
#         return self.max_d
    
#     def get_max_v(self):
#         return self.max_v
    
#     def is_complex_available(self):
#         return self.complex_available

#     def is_zero_velocity(self):
#         return self.assume_zero_velocity
    
#     def where_is_constant_beat_frequency(self, t: np.ndarray, distance: float) -> np.ndarray[bool]:
#         if self.modulation != 'triangle':
#             raise RuntimeError('this method is only valid for triangular modulation')
#         delay = distance * 2 / 3e8
#         time_cushion = 10/self.sample_rate
#         T = self.Tchirp
#         if delay < T:
#             cbf_region1 = ((T-time_cushion)>t)*(t>=(delay+time_cushion))
#             cbf_region2 = ((2*T-time_cushion)>t)*(t>=(T+(delay+time_cushion)))
#         elif (delay >=T) and (delay < 2*T):
#             cbf_region2 = ((T+time_cushion)<t)*(t<=(delay-time_cushion))
#             cbf_region1 = (time_cushion<t)*(t<=(delay-time_cushion-T))
#         else:
#             raise RuntimeError('distance cannot be resolved')
#         return cbf_region1, cbf_region2


# class FMCWMeasurement():
#     def __init__(self, meas_prop: FMCWMeasurementProperties):
#         self.meas_prop = meas_prop
    
#     def assign(self, t, meas):
#         self.meas_time = t
#         self.meas_val = meas

#     def generate_phase(self, ts: np.ndarray, distance: float, velocity: float):

#         T_chirp = self.meas_prop.get_chirp_length()
#         bandwidth = self.meas_prop.get_bandwidth()
#         mode = self.meas_prop.get_modulation_type()
#         center_wavelength = self.meas_prop.get_carrier_wavelength()
#         mixed_coef = self.meas_prop.get_mix_coef()
#         poly_coefs_up, poly_coefs_down = self.meas_prop.get_poly_coefs()

#         delay = distance * 2 / 3e8
#         gamma = bandwidth / T_chirp
#         v = 2*velocity/3e8
#         f0 = 3e8 / center_wavelength - bandwidth / 2

#         if mode == 'sawtooth':

#             ts_w = np.mod(ts-delay-v*ts, T_chirp)
#             phase = 0.5*gamma*(ts_w**2)
#             period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
#             phase_offset = period_num * 0.5*gamma*(T_chirp**2)
#             phase_rx = phase + phase_offset - f0*v*ts

#         elif mode == 'triangle' or mode == 'triangle_phase':

#             ts_w = utils.mirror(ts-delay-v*ts, 0, T_chirp)
#             phase = 0.5*gamma*(ts_w**2)
#             period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
#             phase = phase* (-np.ones(ts.shape))**(period_num)
#             phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(T_chirp**2)
#             phase_offset2 = period_num * 0.5*gamma*(T_chirp**2)
#             phase_rx = phase + phase_offset1 + phase_offset2 - f0*v*ts

#         elif mode == 'sinusoidal':
#             phase_rx = -1/np.pi/2*T_chirp*bandwidth*np.sin(np.pi*(ts-delay-v*ts)/T_chirp) + bandwidth/2*(ts-delay-v*ts) - f0*v*ts

#         elif mode == 'mixed':

#             ts_w = utils.mirror(ts-delay-v*ts, 0, T_chirp)
#             phase = 0.5*gamma*(ts_w**2)
#             period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
#             phase = phase* (-np.ones(ts.shape))**(period_num)
#             phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(T_chirp**2)
#             phase_offset2 = period_num * 0.5*gamma*(T_chirp**2)
#             phase_rx = (phase + phase_offset1 + phase_offset2) * mixed_coef + (-1/np.pi/2*T_chirp*bandwidth*np.sin(np.pi*(ts-delay-v*ts)/T_chirp) + bandwidth/2*(ts-delay-v*ts)) * (1-mixed_coef) - f0*v*ts

#         elif mode =='polynomial':
#             if poly_coefs_up is None or poly_coefs_down is None:
#                 raise ValueError('polynomial coefficients must be specified when selecting pps model')
            
#             up_total = utils._get_polynomial(T_chirp, poly_coefs_up)
#             down_total = utils._get_polynomial(T_chirp, poly_coefs_down)

#             ts_rx = ts-delay-v*ts
#             Tstart = np.floor(ts_rx[0]/T_chirp/2)*T_chirp*2
#             Tend = np.ceil(ts_rx[-1]/T_chirp/2)*T_chirp*2
#             phase_rx = np.zeros((len(ts)))
#             offset = 0
#             for T in np.arange(Tstart, Tend, 2*T_chirp):
#                 interval1 = (T <= ts_rx)*(ts_rx < T+T_chirp)
#                 interval2 = (T+T_chirp <= ts_rx)*(ts_rx <= T+2*T_chirp)
#                 phase_rx[interval1] = offset + utils._get_polynomial(ts_rx[interval1]-T, poly_coefs_up)
#                 phase_rx[interval2] = offset + up_total + utils._get_polynomial(ts_rx[interval2]-T-T_chirp, poly_coefs_down)
#                 offset = offset + up_total + down_total
#             phase_rx = phase_rx - f0*v*ts

#         elif mode == 'smoothstairs':

#             phase = modf.SmoothStairsModulation(self.meas_prop).generate_phase(ts, distance, velocity)
#             phase /= (2*np.pi)

#         return 2*np.pi*phase_rx

#     def generate_reference_phase(self, ts: np.ndarray, distance: float, velocity: float):
#         print("generate_reference_phase is deprecated")
#         return self.generate_phase(ts, distance, velocity)

#     def generate(self, distance: float|ArrayLike, velocity: float|ArrayLike, ts: np.ndarray, 
#                  include_phase_noise: bool|None=None, include_shot_noise : bool|None=None,
#                  return_complex_signal :bool|None=None, 
#                  phase_noise_seed: int|None=None, shot_noise_seed: int|None=None,
#                  return_seeds: bool=False):
        
#         if np.isscalar(distance) and np.isscalar(velocity):
#             n_echo = 1
#             distance = np.array([distance])
#             velocity = np.array([velocity])
#         else:
#             if len(distance) != len(velocity):
#                 raise ValueError('distance and velocity must have same length')
#             n_echo = len(distance)
#             distance = np.array(distance)
#             velocity = np.array(velocity)

#         T_chirp = self.meas_prop.get_chirp_length()
#         bandwidth = self.meas_prop.get_bandwidth()
#         mode = self.meas_prop.get_modulation_type()
#         linewidth = self.meas_prop.get_linewidth()
#         center_wavelength = self.meas_prop.get_carrier_wavelength()
#         mixed_coef = self.meas_prop.get_mix_coef()
#         poly_coefs_up, poly_coefs_down = self.meas_prop.get_poly_coefs()
#         if include_phase_noise is None:
#             include_phase_noise = True
#         if include_shot_noise is None:
#             include_shot_noise = self.meas_prop.include_shot_noise
#         if return_complex_signal is None:
#             return_complex_signal = self.meas_prop.is_complex_available()

#         delay = distance * 2 / 3e8
#         gamma = bandwidth / T_chirp
#         v = 2*velocity/3e8
#         f0 = 3e8 / center_wavelength - bandwidth / 2

#         phase_tx = self.generate_phase(ts, 0, 0)
#         phase_rx = np.zeros((n_echo, len(ts)))
#         for i, (d, v) in enumerate(zip(distance, velocity)):
#             phase_rx[i] = self.generate_phase(ts, d, v)
#         mixed_phase = phase_tx - phase_rx

#         if phase_noise_seed is None:
#             phase_noise_seed = int(np.random.rand(1)*(2**32-1))

#         if shot_noise_seed is None:
#             shot_noise_seed = int(np.random.rand(1)*(2**32-1))

#         if include_phase_noise:
#             np.random.seed(phase_noise_seed)
#             phase_noise = gen_phase_noise(ts, distance, velocity, linewidth)
#             mixed_phase = mixed_phase + phase_noise

#         if return_complex_signal:
#             mixed_signal = np.exp(1j*(mixed_phase))
#         else:
#             mixed_signal = np.cos(mixed_phase)
        
#         R = self.meas_prop.reflectance
#         A = self.meas_prop.detector_effective_area
#         eta = self.meas_prop.detector_effectivity
#         tx_power = self.meas_prop.transmitted_power
#         lo_power = tx_power
#         q = 1.6e-19
#         rx_power = tx_power * A / (np.pi*distance**2) * R
#         rx_interference = 0
#         for i in range(n_echo):
#             rx_interference += rx_power[i]
#             for j in range(0,i):
#                 rx_interference += 2*np.sqrt(rx_power[i]*rx_power[j])*np.cos(phase_rx[i]-phase_rx[j])

#         y1 = 0.5*(lo_power + rx_interference) + np.sqrt(lo_power*rx_power) @ mixed_signal
#         y2 = 0.5*(lo_power + rx_interference) - np.sqrt(lo_power*rx_power) @ mixed_signal
#         if include_shot_noise:
#             np.random.seed(shot_noise_seed)
#             y1 = y1 + np.sqrt(q*eta*np.real(y1)*(np.real(y1)>0))*np.random.randn(*y1.shape)
#             y1 = y1 + 1j*np.sqrt(q*eta*np.imag(y1)*(np.imag(y1)>0))*np.random.randn(*y1.shape)
#             y2 = y2 + np.sqrt(q*eta*np.real(y2)*(np.real(y2)>0))*np.random.randn(*y2.shape)
#             y2 = y2 + 1j*np.sqrt(q*eta*np.imag(y2)*(np.imag(y2)>0))*np.random.randn(*y2.shape)
    
#         u = y1 - y2
#         v = y1 + y2

#         if return_seeds:
#             return_vals = (u, v, phase_noise_seed, shot_noise_seed)
#         else:
#             return_vals = (u, v)

#         return return_vals

#     def get_meas(self):
#         return self.meas_time, self.meas_val
    
#     def get_properties(self):
#         return self.meas_prop


# def estimate_pps_coefficents(t, x, reference_distance, order=3, sample_freq=None):

#     # calculate the wigner distribution
#     Q = order - 1
#     N = x.shape[0]
#     lags = np.array([int(N/(Q))]*(Q-1))
#     bins = np.arange(N)
#     t_offset = t[0]
#     t = t - t_offset
#     if sample_freq is None:
#         sample_freq = 1 / (t[1] - t[0])
#     coef_hat = np.zeros((Q))
#     for i in range(Q):
#         pd = np.copy(x)
#         for l in lags:
#             padded_x = np.pad(pd, (0, N), 'constant', constant_values=0)
#             pd = padded_x[l:l+N] * pd
#         freq = max_periodogram(pd)
#         coef_hat[-(i+1)] = freq / (factorial(Q-i)*np.prod(lags)) * sample_freq**(Q-i)
#         if i < (Q-1):
#             x = x * np.exp(-1j*2*np.pi*(t**(Q-i))*coef_hat[-(i+1)])
#             lags = lags[:-1]

        
#     ref_tau = reference_distance * 2 / 3e8

#     coef_transformation = pascal(order+1, kind='lower')[1:, 1:] - np.identity(order)
#     coef_transformation = coef_transformation.T
#     coef_transformation = coef_transformation[:-1,1:]
#     coef_transformation = coef_transformation.astype(float)
#     for i in range(order-1):
#         for j in range(i, order-1):
#             # coef_transformation[i,j] = coef_transformation[i,j] * ((+t_offset)**(j-i+1)-(+t_offset-ref_tau)**(j-i+1))
#             coef_transformation[i,j] = coef_transformation[i,j] * ((+t_offset)**(j-i+1)-(+t_offset-ref_tau)**(j-i+1))
#     coef_hat = np.linalg.inv(coef_transformation) @ coef_hat
#     coef_hat = np.concatenate([[0], coef_hat])
#     # coef_hat[1] = coef_hat[1] - gamma

#     return coef_hat

def hilbert_transform(signal):
    assert signal.dtype == float
    Fx = fft(signal)
    idx = np.arange(len(Fx))
    N = len(Fx)
    pos_freq = ( idx < (N-N//2) ) * ( idx > 0)
    neg_freq = ( idx >= (N-N//2) )
    Fx[pos_freq] = Fx[pos_freq] * -1j
    Fx[neg_freq] = Fx[neg_freq] * 1j
    Fx[0] = 0
    return signal + 1j* np.real(ifft(Fx))

# def pps_compute_down_coefs(up_coefs, Tchirp):
#     tmp_down_coefs = np.zeros_like(up_coefs)
#     for k, up_coef in enumerate(up_coefs):
#         if k % 2 == 0:
#             tmp_down_coefs[k] = up_coef
#         else:
#             tmp_down_coefs[k] = -up_coef

#     order = len(up_coefs)
#     coef_transformation = pascal(order+1, kind='lower')[1:, 1:]
#     coef_transformation = coef_transformation.T
#     coef_transformation = coef_transformation.astype(float)
#     for i in range(order):
#         for j in range(i+1, order):
#             coef_transformation[i,j] = coef_transformation[i,j] * ((-Tchirp)**(j-i))

#     down_coefs = coef_transformation @ tmp_down_coefs
#     return down_coefs

def import_meas_prop_from_config(path: str, configname: str) -> ChirpGeneratorConfig:

    f = open(path)
    config_sets = json.load(f)
    config = config_sets[configname]
    modulation = config["modulation"]
    T = config["T"]
    B = config["B"]
    sample_rate = config["sample_rate"]
    linewidth = config["linewidth"]
    carrier_wavelength = config["lambd_c"]
    reflectance = config["reflectance"]
    detector_effectivity = config["detector_effectivity"] # A/W
    transmitted_power = config["transmitted_power"] # W
    detector_effective_area = config["detector_effective_area"] # m^2
    complex_available = config["complex_available"]
    include_shot_noise = config["include_shot_noise"]
    f.close()

    meas_prop = FMCWMeasurementProperties(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=True, modulation=modulation,
                                          include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, reflectance=reflectance,
                                          detector_effectivity=detector_effectivity, detector_effective_area=detector_effective_area,
                                          complex_available=complex_available)
    
    return meas_prop

def import_from_meas_data(path: str):
    meas_data = np.load(path)
    distance_arr = meas_data['distance']
    if 'velocity' in meas_data.keys():
        velocity_arr = meas_data['velocity']
        velocity_arr_exists = True
    else:
        velocity_arr_exists = False
    modulation = meas_data['modulation']
    T = meas_data['Tchirp']
    B = meas_data['bandwidth']
    sample_rate = meas_data['sample_rate']
    linewidth = meas_data['linewidth']
    carrier_wavelength = meas_data['lambd_c']
    reflectance = meas_data['reflectance']
    detector_effectivity = meas_data['detector_effectivity'] # A/W
    transmitted_power = meas_data['transmitted_power'] # W
    detector_effective_area = meas_data['detector_effective_area'] # m^2
    complex_available = meas_data['complex_available']
    include_shot_noise = meas_data['include_shot_noise']
    measurement_arr = meas_data['measurement']
    t = np.arange(0, measurement_arr.shape[3], 1) / sample_rate
    n_simulation = measurement_arr.shape[1]
    if velocity_arr_exists:
        assume_zero_velocity = False
    else:
        assume_zero_velocity = True
    # create fmcw measurement system
    meas_prop = ChirpGeneratorConfig(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=assume_zero_velocity, 
                                          modulation=modulation, include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, 
                                          reflectance=reflectance, detector_effectivity=detector_effectivity, 
                                          detector_effective_area=detector_effective_area, complex_available=complex_available)
    if velocity_arr_exists:
        return_vals = (n_simulation, distance_arr, velocity_arr, measurement_arr, meas_prop)
    else:
        return_vals = (n_simulation, distance_arr, measurement_arr, meas_prop)
    return return_vals

def import_sim_data_from_meas_data(path: str):
    meas_data = np.load(path)
    distance_arr = meas_data['distance']
    if 'velocity' in meas_data.keys():
        velocity_arr = meas_data['velocity']
        velocity_arr_exists = True
    else:
        velocity_arr_exists = False
    modulation = str(meas_data['modulation'])
    T = float(meas_data['Tchirp'])
    B = float(meas_data['bandwidth'])
    sample_rate = float(meas_data['sample_rate'])
    linewidth = float(meas_data['linewidth'])
    carrier_wavelength = float(meas_data['lambd_c'])
    reflectance = float(meas_data['reflectance'])
    detector_effectivity = float(meas_data['detector_effectivity']) # A/W
    transmitted_power = float(meas_data['transmitted_power']) # W
    detector_effective_area = float(meas_data['detector_effective_area']) # m^2
    complex_available = bool(meas_data['complex_available'])
    include_shot_noise = meas_data['include_shot_noise']
    measseeds_arr = meas_data['measseeds']
    n_cycle = int(meas_data['n_cycle'])
    n_simulation = int(meas_data['n_simulation'])
    t = np.arange(0, 2*T*n_cycle, 1) / sample_rate
    if velocity_arr_exists:
        assume_zero_velocity = False
    else:
        assume_zero_velocity = True
    # create fmcw measurement system
    meas_prop = ChirpGeneratorConfig(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=assume_zero_velocity, 
                                          modulation=modulation, include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, 
                                          reflectance=reflectance, detector_effectivity=detector_effectivity, 
                                          detector_effective_area=detector_effective_area, complex_available=complex_available)
    sim_data = dict()
    sim_data['n_simulation'] = int(n_simulation)
    sim_data['distance'] = distance_arr
    if velocity_arr_exists:
        sim_data['velocity'] = velocity_arr
    sim_data['measseeds'] = measseeds_arr
    sim_data['meas_prop'] = meas_prop
    sim_data['n_cycle'] = int(n_cycle)

    return sim_data

# highest order first
def compute_polynomial_derivative(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs[-2::-1]):
        x = x + (i+1)*(t**(i))*c
    return x