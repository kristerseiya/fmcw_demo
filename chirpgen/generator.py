
import numpy as np
from scipy.fft import rfft, irfft, fftfreq
import warnings
import re

from . import modulation
from .config import ChirpGeneratorConfig
import utils
from typing import Literal

def generate_noise_from_acf_(acf):

    n0 = len(acf)
    acf_symmetric = np.concatenate([acf, acf[::-1][1:-1]])

    n = len(acf_symmetric)
    
    # 1. Compute PSD
    psd = np.fft.fft(acf_symmetric)
    psd = np.maximum(np.real(psd), 0) # Ensure positivity
    
    # 2. Generate frequency domain noise
    # We use complex normal, but we need to scale so the expected 
    # power matches the PSD magnitude.
    # The 1/sqrt(2) keeps the total variance of the complex number at 1.
    # white_noise = (np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)) #/ np.sqrt(2)
    # white_noise[1:] = 0.5 * (white_noise[:0:-1] + white_noise[1:])
    white_noise = np.random.normal(0, 1, n)
    white_noise = np.fft.fft(white_noise, norm="ortho")
    # 3. Apply the "filter" (PSD)
    # We multiply by sqrt(n) because the IFFT divides by n.
    # This keeps the variance of the output equal to acf[0].
    colored_noise_fft = white_noise * np.sqrt(psd) #* np.sqrt(n)
    
    # 4. Transform back
    noise = np.fft.ifft(colored_noise_fft, norm="ortho")
    
    return np.real(noise[:n0])

def generate_noise_from_acf(acf):
    n0 = len(acf)
    acf_symmetric = np.concatenate([acf, acf[::-1][1:-1]])
    n = len(acf_symmetric)
    psd = np.fft.fft(acf_symmetric, norm="forward")
    psd = np.maximum(np.real(psd), 0) # Ensure positivity
    white_noise1 = np.random.normal(0, 1/np.sqrt(2), n0)
    white_noise2 = np.random.normal(0, 1/np.sqrt(2), n0-1)
    white_noise_symmetric = np.concatenate([ [white_noise1[0],], white_noise1[1:]+1j*white_noise2, white_noise1[::-1][1:-1]-1j*white_noise2[::-1][1:]])
    colored_noise_fft = white_noise_symmetric * np.sqrt(psd) #* np.sqrt(n)
    noise = np.fft.ifft(colored_noise_fft, norm="forward")
    noise = np.real(noise)
    return noise[:n0]

def generate_noise_from_psd(psd):
    n = len(psd)
    if n % 2 == 0:
        n0 = n//2 + 1
        white_noise1 = np.random.normal(0, 1/np.sqrt(2), n0)
        white_noise2 = np.random.normal(0, 1/np.sqrt(2), n0-1)
        white_noise_symmetric = np.concatenate([ [white_noise1[0],], white_noise1[1:]+1j*white_noise2, white_noise1[::-1][1:-1]-1j*white_noise2[::-1][1:]])
    else:
        n0 = n//2 + 1
        white_noise1 = np.random.normal(0, 1/np.sqrt(2), n0)
        white_noise2 = np.random.normal(0, 1/np.sqrt(2), n0-1)
        white_noise_symmetric = np.concatenate([ [white_noise1[0],], white_noise1[1:]+1j*white_noise2, white_noise1[::-1][:-1]-1j*white_noise2[::-1]])
    colored_noise_fft = white_noise_symmetric * np.sqrt(psd)
    noise = np.fft.ifft(colored_noise_fft, norm="forward")
    noise = np.real(noise)
    return noise

def generate_phase_noises(ts:np.ndarray, 
                          delays:float|np.ndarray,
                          noise_type:str="white",
                          generation_period:None|float=None,
                          var:float=1,
                          acf_fun=None,
                          psd_fun=None,
                          start_time="auto",
                          return_original=False,
                          ) -> np.ndarray[float]:

    # ts_diff = np.diff(ts)
    # avg_sample_period = np.mean(ts_diff)
    # min_sample_period = np.min(ts_diff)
    del_t = ts[1] - ts[0]
    sample_rate = 1/del_t

    if generation_period is None:
        generation_period = del_t/20
    
    if generation_period > del_t:
        warnings.warn("geneartion period is larger than minimum sampling period")
    
    if np.isscalar(delays):
        delays = np.array([delays])
    else:
        delay_sorted = np.sort(delays)
        min_delay_diff = np.min(np.diff(delay_sorted))

        if generation_period > del_t:
            warnings.warn("minimum delay difference is smaller than generation period")

    if start_time == "auto":
        gen_ts = np.arange(ts[0]-np.amax(delays), ts[-1]+generation_period, generation_period)
    else:
        assert(start_time <= (ts[0]-np.amax(delays)))
        gen_ts = np.arange(start_time, ts[-1]+generation_period, generation_period)

    if noise_type == "white":
        gen_ts_diff = np.diff(gen_ts)
        noise = np.random.randn(len(gen_ts_diff))
        gen_noise = np.cumsum(noise*np.sqrt(gen_ts_diff*var))
        random_offset = np.random.rand() * 2*np.pi * 0
        gen_noise = np.concatenate([[random_offset], gen_noise+random_offset])

    elif bool(re.match("rect\d+$", noise_type)):
        n_coef = int(noise_type[4:])
        rescaled_var = var * len(gen_ts) / (2*n_coef+1)
        gen_ts_diff = np.diff(gen_ts)
        noise = np.random.randn(len(gen_ts_diff))
        gen_noise = np.cumsum(noise*np.sqrt(gen_ts_diff*rescaled_var))
        random_offset = np.random.rand() * 2*np.pi  * 0
        gen_noise = np.concatenate([[random_offset], gen_noise+random_offset])
        gen_noise_fc = rfft(gen_noise, n=len(gen_noise))
        gen_noise_fc[n_coef:] = 0
        gen_noise = irfft(gen_noise_fc, n=len(gen_noise))

    elif noise_type == "stationary":
        acf = acf_fun(gen_ts-np.amin(gen_ts)) * var * del_t
        gen_noise = generate_noise_from_acf(acf)

    elif noise_type == "stationary_freq_from_acf":
        acf = acf_fun(gen_ts-np.amin(gen_ts))
        acf = acf / acf_fun(0) * var 
        gen_noise = generate_noise_from_acf(acf)
        gen_noise = np.cumsum(gen_noise) / len(gen_noise) * (gen_ts[-1] - gen_ts[0])

    elif noise_type == "stationary_freq_from_psd":
        # psd = psd_fun(np.arange(0, 0.5+1/len(gen_ts), 1/len(gen_ts))[:len(gen_ts)])
        psd = psd_fun(fftfreq(len(gen_ts))/generation_period)
        psd = psd / np.sum(psd) * var
        gen_noise = generate_noise_from_psd(psd)
        gen_noise = np.cumsum(gen_noise) / len(gen_noise) * (gen_ts[-1] - gen_ts[0])

    elif noise_type == "stationary_phase_from_acf":
        acf = acf_fun(gen_ts-np.amin(gen_ts))
        acf = acf / acf_fun(0) * var 
        gen_noise = generate_noise_from_acf(acf)

    elif noise_type == "stationary_phase_from_psd":
        psd = psd_fun(np.arange(0, 0.5+1/len(gen_ts), 1/len(gen_ts))[:len(gen_ts)])
        psd = psd / np.sum(psd) * var
        gen_noise = generate_noise_from_psd(psd)

    phase_noises = np.zeros((len(delays), len(ts)))
    for i, delay in enumerate(delays):
        phase_noises[i] = np.interp(ts-delay, gen_ts, gen_noise)
    
    if len(delays) == 1:
        phase_noises = phase_noises[0]

    if return_original:
        return phase_noises, gen_ts, gen_noise
    
    return phase_noises

class ChirpGenerator():
    def __init__(self, config: ChirpGeneratorConfig):
        self.config = config
        self.modulation_func = modulation.get_modulation(config)
        self.curr_phase_noise = None

    def generate(self, 
                 ts:np.ndarray,
                 distance:float|np.ndarray, 
                 velocity:float|np.ndarray=0, 
                 reflection_coef:float|np.ndarray=1,
                 return_phasenoise: bool=False,
                 same_phasenoise: bool=False):
        
        if isinstance(distance, np.ndarray):
            if np.isscalar(velocity):
                velocity = velocity * np.ones_like(distance)
            if np.isscalar(reflection_coef):
                reflection_coef = reflection_coef * np.ones_like(distance)
        elif np.isscalar(distance) and np.isscalar(velocity) and np.isscalar(reflection_coef):
            distance = np.array([distance])
            velocity = np.array([velocity])
            reflection_coef = np.array([reflection_coef])

        bandwidth = self.config.get_bandwidth()
        sample_rate = self.config.sample_rate
        Trep = self.config.Trep
        dechirp = self.config.dechirp

        additive_noise_stdv = self.config.additive_noise_stdv
        phase_noise_stdv = self.config.phase_noise_stdv
        distance_fall_off = self.config.distance_fall_off
        return_complex_signal = self.config.complex_valued
        generation_period = self.config.phase_noise_gen_delta_t
        
        A = self.config.detector_effective_area
        eta = self.config.detector_effectivity
        tx_power = self.config.transmitted_power
        q = 1.6e-19
        if distance_fall_off:
            rx_power = tx_power * A / (np.pi*distance**2) * reflection_coef
        else:
            rx_power = tx_power*reflection_coef
        
        distance0 = np.concatenate([[0], distance])
        velocity0 = np.concatenate([[0], velocity])

        waveforms = self.modulation_func.generate_waveform(ts, distance0, velocity0)


        if (phase_noise_stdv > 0) or return_phasenoise:
            delays = distance0 * 2 / 3e8
            if same_phasenoise and (self.curr_phase_noise is not None):
                gen_t, gen_pn = self.curr_phase_noise
                phase_noise = np.zeros((len(delays), len(ts)))
                for i, delay in enumerate(delays):
                    phase_noise[i] = np.interp(ts-delay, gen_t, gen_pn)
                if len(delays) == 1:
                    phase_noise = phase_noise[0]
            else:
                if generation_period is None:
                    generation_period = 1/max(sample_rate, bandwidth) / 20
                phase_noise, gen_t, gen_pn = generate_phase_noises(ts, delays, noise_type=self.config.phase_noise_type, 
                                                    generation_period=generation_period, var=phase_noise_stdv**2,
                                                    acf_fun=self.config.acf_fun,
                                                    psd_fun=self.config.psd_fun,
                                                    start_time=ts[0]-Trep,
                                                    return_original=True)
                self.curr_phase_noise = (gen_t, gen_pn)
            
            waveforms = waveforms*np.exp(1j*phase_noise)
        
        tx_signal = np.sqrt(tx_power) * waveforms[0]
        rx_signal = np.sqrt(rx_power) @ waveforms[1:]
        

        if not dechirp:

            if return_complex_signal:
                y = eta * rx_signal
            else:
                y = eta * np.real(rx_signal)
                
            if distance_fall_off:
                y = y + np.sqrt(q*np.real(y)*(np.real(y)>0))*np.random.randn(*y.shape)
                y = y + 1j*np.sqrt(q*np.imag(y)*(np.imag(y)>0))*np.random.randn(*y.shape)

        else:
            # rx_interference = 0
            # for i in range(n_echo):
            #     rx_interference += rx_power[i]
            #     for j in range(0,i):
            #         rx_interference += 2*np.sqrt(rx_power[i]*rx_power[j])*np.cos(phase_rx[i]-phase_rx[j])
                    
            # rx_signal * np.conj(rx_signal)
            
            crossterm = tx_signal * np.conj(rx_signal)
            autoterm = 0.5*( np.square(np.abs(tx_signal)) + np.square(np.abs(rx_signal)) )*(1+1j)
            y1 = eta*(autoterm + crossterm)
            y2 = eta*(autoterm - crossterm)
            
            if not return_complex_signal:
                y1 = np.real(y1)
                y2 = np.real(y2)
            
            # rx_interference = 0
            # for i in range(n_echo):
            #     rx_interference += rx_power[i]
            #     for j in range(0,i):
            #         rx_interference += 2*np.sqrt(rx_power[i]*rx_power[j])*np.cos(phase_rx[i]-phase_rx[j])

            # if return_complex_signal:
            #     y1 = 0.5*eta*(lo_power + rx_interference)*(1+1j) + eta*np.sqrt(lo_power*rx_power) @ np.exp(1j*(phase_rx))
            #     y2 = 0.5*eta*(lo_power + rx_interference)*(1+1j) - eta*np.sqrt(lo_power*rx_power) @ np.exp(1j*(phase_rx))
            # else:
            #     y1 = 0.5*eta*(lo_power + rx_interference) + eta*np.sqrt(lo_power*rx_power) @ np.cos(phase_rx)
            #     y2 = 0.5*eta*(lo_power + rx_interference) - eta*np.sqrt(lo_power*rx_power) @ np.cos(phase_rx)
            
            

            if distance_fall_off:
                y1 = y1 + np.sqrt(q*np.real(y1)*(np.real(y1)>0))*np.random.randn(*y1.shape)
                y1 = y1 + 1j*np.sqrt(q*np.imag(y1)*(np.imag(y1)>0))*np.random.randn(*y1.shape)
                y2 = y2 + np.sqrt(q*np.real(y2)*(np.real(y2)>0))*np.random.randn(*y2.shape)
                y2 = y2 + 1j*np.sqrt(q*np.imag(y2)*(np.imag(y2)>0))*np.random.randn(*y2.shape)

            y = y1 - y2


        y = y + additive_noise_stdv * np.random.randn(len(y))
        if return_complex_signal:
            y = y + additive_noise_stdv * 1j * np.random.randn(len(y))

        if return_phasenoise:
            return y, (gen_t, gen_pn)

        return y

