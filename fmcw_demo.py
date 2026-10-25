
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import maximum_filter1d
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import cvxpy as cp
from scipy.linalg import khatri_rao
import chirpgen
import time_freq
import filters
from utils import upsample
import utils
import cs_algo

Tchirp = 1e-6
Trep = 2e-6
sample_rate = 1e9
bandwidth = 3e9
K = 3

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                        Trep=Trep, 
                                        sample_rate=sample_rate,
                                        bandwidth=bandwidth,
                                        modulation="triangle",
                                        generation_type="continuous",
                                        lambd_c=1e-3,
                                        dechirp=True,
                                        phase_noise_stdv=0,#3e8,
                                        additive_noise_stdv=0.1,
                                        distance_fall_off=False,
                                        phase_noise_type="stationary_freq_from_psd",
                                        psd_fun=psd_fun,
                                        complex_valued=True,
                                        transmitted_power=1,
                                        )


generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep, 1/sample_rate)
distance = np.random.rand(K) * Trep * 3e8 / 2
distance = np.array([31,])
refl = np.array([1,])
y = generator.generate(t, distance, reflection_coef=refl)

M = np.ceil(bandwidth/sample_rate).astype(int)*3
t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
modf = chirpgen.modulation.get_modulation(config)
tx = modf.generate_waveform(t_fine)

# plt.subplot(1,2,1)
# ff, tt, xx = time_freq.short_time_fourier_transform(tx, 64, 16, sample_rate=sample_rate, zero_pad=512-32)
# plt.pcolormesh(tt, ff, np.abs(xx), cmap="inferno")
# plt.xlabel("time")
# plt.ylabel("frequency")
tx = bandwidth * utils.mirror(t, 0, Tchirp)
rx = bandwidth * utils.mirror(t-distance[0]*2/3e8, 0, Tchirp)
plt.figure(figsize=(16, 8))
plt.subplot(2,2,1)
plt.plot(t, tx, color="green", label="TX")
plt.subplot(2,2,2)
plt.plot(t, rx, color="red", label="RX")
plt.subplot(2,2,3)
plt.plot(t, tx, color="green", label="TX")
plt.plot(t, rx, color="red", label="RX")
plt.subplot(2,2,4)
ff, tt, xx = time_freq.short_time_fourier_transform(np.real(y), 64, 16, sample_rate=sample_rate, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.abs(xx), cmap="inferno")
plt.xlabel("time")
plt.ylabel("frequency")
plt.tight_layout()
plt.show()