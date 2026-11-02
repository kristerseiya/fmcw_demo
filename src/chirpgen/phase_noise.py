
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike

def gen_phase_noise(ts: ArrayLike, 
                    distance: float|ArrayLike, 
                    velocity: float|ArrayLike, 
                    linewidth: float,) -> np.ndarray[float]:

    if isinstance(distance, float):
        distance = np.array([distance])
        velocity = np.array([velocity])
    else:
        if len(distance) != len(velocity):
            raise ValueError("Distance and velocity must have same length")
        distance = np.array(distance)
        velocity = np.array(velocity)

    n_echo = len(distance)
    N_t = len(ts)

    tt = ts -  2*(np.reshape(distance,(-1,1))+np.reshape(velocity,(-1,1))*ts)/3e8

    ts_all = np.concatenate([ts, *[tt[i] for i in range(tt.shape[0])]])
    sort_idx = np.argsort(ts_all)
    ts_all = ts_all[sort_idx]

    phase_noise = np.zeros_like(ts_all)
    # phase_noise[0] = np.random.rand() * 2*np.pi
    
    for i in range(1, 2*N_t):
        delay = ts_all[i] - ts_all[i-1]
        phase_noise[i] = phase_noise[i-1] + np.random.normal(0, scale=np.sqrt(np.abs(delay)*linewidth*2*np.pi))

    phase_noise_mix = np.zeros((n_echo, N_t))
    tx_idx = (sort_idx >= 0) * (sort_idx < N_t)
    phase_noise_tx = phase_noise[tx_idx]
    for n in range(n_echo):
        rx_idx = (sort_idx >= (n+1)*N_t) * (sort_idx < (n+2)*N_t)
        phase_noise_rx = phase_noise[rx_idx]
        phase_noise_mix[n] = phase_noise_tx - phase_noise_rx

    if n_echo == 1:
        phase_noise_mix = phase_noise_mix[0]

    return phase_noise_mix
