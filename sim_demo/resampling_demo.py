
import numpy as np
import matplotlib.pyplot as plt
import chirpgen
import time_freq
import utils
import algo.resample
from scipy.fft import fft

# plt.rcParams["image.cmap"] = "magma"
plt.rcParams["image.cmap"] = "inferno"

Tchirp = 10e-6
Trep = 20e-6
sample_rate = 1e9
bandwidth = 10e9
K = 3

ts = np.linspace(-1,2,10000)
ts_w = utils.mirror(ts, 0, 0.5)*2
M = 100
ifq = np.pad(ts_w, (M, M), mode="edge")
ifq = np.convolve(ifq, np.ones((2*M+1,))/(2*M+1), mode="valid")

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                        Trep=Trep, 
                                        sample_rate=sample_rate,
                                        bandwidth=bandwidth,
                                        modulation="custom",
                                        modulation_data=ifq,
                                        generation_type="continuous",
                                        lambd_c=1e-3,
                                        dechirp=True,
                                        phase_noise_stdv=bandwidth*0.003,
                                        additive_noise_stdv=0,
                                        distance_fall_off=False,
                                        phase_noise_type="stationary_freq_from_psd",
                                        psd_fun=psd_fun,
                                        complex_valued=True,
                                        phase_noise_gen_delta_t=1/max(sample_rate, bandwidth)/20,
                                        transmitted_power=1,
                                        )


generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep, 1/sample_rate)
distance = np.array([17, 21.5, 26])
# refl = [0.4, 1, 0.1]
refl = [1, 1, 1]
# refl = [0.001, 1, 0.001]
aux_distance = 21.5
y = generator.generate(t, distance, reflection_coef=refl) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.5
aux = generator.generate(t, aux_distance, reflection_coef=1, same_phasenoise=True) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.1
y = np.real(y)
aux = np.real(aux)

cbf1_start, cbf1_end = int(len(y)*0.1), int(len(y)*0.4)
aux_cbf1 = aux[cbf1_start:cbf1_end]
y_cbf1 = y[cbf1_start:cbf1_end]
y_cbf1_new = algo.resample.resampling_by_linerized_phase(aux_cbf1, y_cbf1)
# y = np.real(y)
# print(len(y))

ff1, tt1, xx1 = time_freq.short_time_fourier_transform(y, window=64, stride=32, sample_rate=sample_rate, zero_pad=512-64)
ff2, tt2, xx2 = time_freq.short_time_fourier_transform(aux, window=64, stride=32, sample_rate=sample_rate, zero_pad=512-64)
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.pcolormesh(tt1, ff1, np.abs(xx1))
plt.subplot(1,2,2)
plt.pcolormesh(tt2, ff2, np.abs(xx2))
plt.tight_layout()

ff3, tt3, xx3 = time_freq.short_time_fourier_transform(y_cbf1, window=64, stride=32, sample_rate=sample_rate, zero_pad=512-64)
ff4, tt4, xx4 = time_freq.short_time_fourier_transform(y_cbf1_new, window=64, stride=32, sample_rate=sample_rate, zero_pad=512-64)
plt.figure(figsize=(12, 4))
plt.subplot(2,2,1)
plt.pcolormesh(tt3, ff3, np.abs(xx3))
plt.subplot(2,2,2)
plt.pcolormesh(tt4, ff4, np.abs(xx4))
plt.subplot(2,2,3)
plt.plot(np.abs(fft(y_cbf1)))
plt.subplot(2,2,4)
plt.plot(np.abs(fft(y_cbf1_new)))

plt.show()