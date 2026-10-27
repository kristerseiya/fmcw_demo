
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
from filters import SignalModel, construct_matchfilter
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
                                        additive_noise_stdv=0.00001,
                                        distance_fall_off=False,
                                        phase_noise_type="stationary_freq_from_psd",
                                        psd_fun=psd_fun,
                                        complex_valued=True,
                                        transmitted_power=1,
                                        )


generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep, 1/sample_rate)
distance = np.random.rand(K) * Trep * 3e8 / 2
refl = [1, 0.5, 0.5]
y = generator.generate(t, distance,reflection_coef=refl)
# y = np.real(y)

model = SignalModel(config, model_type="config")
M = np.ceil(bandwidth/sample_rate).astype(int)*10
normalized_delays, amplitude, mf0 = algo.sparse.matching_pursuit(model, t, y, M, refine=False, 
                                                            orthogonalize=False, n_iter=3, 
                                                            get_first_mf_response=True)

lasso_amp = algo.sparse.basis_pursuit(model, t, y, M, lambd=10, gamma=1, n_iter=100, regularization="l1")

plt.subplot(121)
f, t, x = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
plt.pcolormesh(t, f, np.abs(x))

plt.subplot(122)
plt.stem(normalized_delays, np.abs(amplitude), linefmt='g--', markerfmt='o', basefmt=' ', label="omp")
plt.plot(np.arange(len(mf0))/(len(mf0)/len(y)), mf0, label="omp", color="green")
plt.plot(np.arange(len(lasso_amp))/M,np.abs(lasso_amp), zorder=3, marker='o', markersize=1, label="bp", color="blue")

plt.vlines((distance*2/3e8*sample_rate), ymin=0, ymax=1, linestyle="-", color="red", zorder=1, label="true")
plt.scatter((distance*2/3e8*sample_rate), np.abs(refl), marker="*",  color="red", zorder=5, label="true")

print(len(y))
plt.legend()
plt.title("Modulus Amplitude and Time Delay Estimates")
plt.show()