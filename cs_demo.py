
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
import algo.sparse
from utils import upsample

Tchirp = 1e-6
Trep = 2e-6
sample_rate = 1e9
bandwidth = 10e9
K = 3

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                        Trep=Trep, 
                                        sample_rate=sample_rate,
                                        bandwidth=bandwidth,
                                        modulation="sinusoidal",
                                        generation_type="continuous",
                                        lambd_c=1e-3,
                                        dechirp=False,
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
refl = [1, 0.1, 0.01]
y = generator.generate(t, distance,reflection_coef=refl)
# y = np.real(y)

normalized_delays, amplitude = algo.sparse.offgrid_orthogonal_matching_pursuit(config, t, y, K)

# lasso_amp = basis_pursuit_faster(config, t, y, 3, 30)

M = np.ceil(bandwidth/sample_rate).astype(int)
t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
mf = filters.construct_matchfilter(t_fine, config, normalize=False)
lasso_amp = algo.sparse.basis_pursuit(mf, y, lambd=10, gamma=1, iter=100, regularization="l1")

plt.subplot(121)
f, t, x = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
plt.pcolormesh(t, f, np.abs(x))

plt.subplot(122)
plt.stem(normalized_delays, np.abs(amplitude), linefmt='g--', markerfmt='o', basefmt=' ', label="omp")
plt.plot(np.arange(len(lasso_amp))/len(lasso_amp),np.abs(lasso_amp), zorder=3, marker='o', markersize=1, label="bp", color="blue")

plt.vlines((distance[:2]*2/3e8/Trep), ymin=0, ymax=np.abs(refl[:2]), linestyle="-", color="red", zorder=1, label="true")
plt.vlines((distance[2]*2/3e8/Trep), ymin=0, ymax=np.abs(refl[2]), linestyle="-", color="red", zorder=5, label="true")

print(len(y))
plt.legend()
plt.title("Modulus Amplitude and Time Delay Estimates")
plt.show()