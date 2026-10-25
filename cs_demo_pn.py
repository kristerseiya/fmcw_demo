
import numpy as np
import matplotlib.pyplot as plt
import chirpgen
import time_freq
import filters
import algo.sparse
from complex_sparse_viz import visualize

Tchirp = 1e-6
Trep = 2e-6
sample_rate = 1e9
bandwidth = 3e9

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
                                        phase_noise_stdv=bandwidth*0.01,
                                        additive_noise_stdv=0.1,
                                        distance_fall_off=False,
                                        phase_noise_type="stationary_freq_from_psd",
                                        psd_fun=psd_fun,
                                        complex_valued=True,
                                        phase_noise_gen_delta_t=1/max(sample_rate, bandwidth)/20,
                                        transmitted_power=1,
                                        )


generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep, 1/sample_rate)
distance = np.random.rand(3) * Trep * 3e8 / 2
refl = [1, 0.1, 0.01]
y = generator.generate(t, distance, reflection_coef=refl)

normalized_delays, amplitude = algo.sparse.offgrid_orthogonal_matching_pursuit(config, t, y, 10)

M = np.ceil(bandwidth/sample_rate).astype(int)*5
t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
mf = filters.construct_matchfilter(t_fine, config, normalize=False)
ifq = np.angle(mf[1:]*np.conj(mf[:-1]))/(2*np.pi)
lasso_amp = algo.sparse.basis_pursuit(mf, y, lambd=3e-2, rho=1, iter=1000)
lasso_amp2raw = algo.sparse.bartlett_basis_pursuit(mf, y, lambd=10, 
                                                   window_len=300, stride=100, rho=1000,
                                                   window="blackmanharris", iter=1000)

lasso_amp2 = np.mean(np.abs(lasso_amp2raw), axis=0)

plt.figure(figsize=(12, 4))
plt.subplot(121)
f, t, x = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
plt.pcolormesh(t, f, np.abs(x))
plt.title("STFT")

plt.subplot(122)
plt.stem(normalized_delays*Trep/2*3e8, np.abs(amplitude), linefmt='g--', markerfmt='o', basefmt=' ', label="omp")
plt.plot(np.arange(len(lasso_amp))/len(lasso_amp)*Trep/2*3e8,np.abs(lasso_amp), zorder=2, marker='o', markersize=1, label="bp", color="blue")
plt.plot(np.arange(len(lasso_amp2))/len(lasso_amp2)*Trep/2*3e8,np.abs(lasso_amp2), zorder=3, marker='o', markersize=1, label="bbp", color="orange")

plt.vlines((distance), ymin=0, ymax=1, linestyle="-", color="red", zorder=1, label="true")
plt.scatter((distance), np.abs(refl), marker="*", c="red", zorder=5)
plt.yscale("log")
plt.ylim(1e-3, 3)
plt.legend()

plt.title("Modulus Amplitude and Time Delay Estimates")

plt.show()