
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import chirpgen
import time_freq
from filters import SignalModel
import algo.sparse
from complex_sparse_viz import visualize
import fmcw_aux
import algo.filter

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['Set1'].colors)
mpl.rcParams['image.cmap'] = 'inferno'
# plt.set_cmap('inferno')
set_colors = plt.colormaps['Set1'].colors

algo.sparse.VERBOSE = True

def _normalize(x):
    return x / np.amax(x)

Tchirp = 2e-6
Trep = 4e-6
sample_rate = 1e9

bandwidth = 3e9

def psd_fun(f):
    return (np.abs(f)<0.01*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                        Trep=Trep, 
                                        sample_rate=sample_rate,
                                        bandwidth=bandwidth,
                                        modulation="sinusoidal",
                                        generation_type="continuous",
                                        lambd_c=1e-3,
                                        dechirp=True,
                                        phase_noise_stdv=bandwidth*0.02,
                                        additive_noise_stdv=0,
                                        distance_fall_off=False,
                                        phase_noise_type="stationary_freq_from_psd",
                                        psd_fun=psd_fun,
                                        complex_valued=True,
                                        phase_noise_gen_delta_t=1/max(sample_rate, bandwidth)/20,
                                        transmitted_power=1,
                                        )

generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep*3, 1/sample_rate)
distance = np.array([14.51, 20.08, 24.3])
# distance = np.array([14.51, 21.59, 24.36])
aux_distance = 21.59
refl = [0.1, 1, 0.01]
y = generator.generate(t, distance, reflection_coef=refl) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.1
aux = generator.generate(t, aux_distance, reflection_coef=1, same_phasenoise=True) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.1
y = np.real(y)
aux = np.real(aux)
zero_crossings = [(0,400), (1800,2200), (3800,4200), (5800,6200), (7800,8200), (9800,10200)]


# plt.figure(figsize=(16,12))
# plt.subplot(331)
ff, tt, xx = time_freq.short_time_fourier_transform(aux, 32, 8, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.log10(np.abs(xx)))
plt.title("auxiliary signal")
plt.show()