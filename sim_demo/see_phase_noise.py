
import numpy as np
import chirpgen
import matplotlib.pyplot as plt
from chirpgen.generator import generate_noise_from_psd
from scipy.fft import fftfreq

sample_rate = 1e9
ts = np.arange(0, 3000)/sample_rate
phase_noise_stdv = 4
awgn_stdv = 1
delays = np.arange(1, 500)
middle = slice(500, 2500)

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)
def generate_diff_matrix(delay):
    

psd = psd_fun(fftfreq(len(ts))*sample_rate)
psd = psd / np.sum(psd) * phase_noise_stdv**2
gen_noise = generate_noise_from_psd(psd)

for d in delays:
    delayed_gen_noise = np.roll(gen_noise, (d,))
    obs_noise = (gen_noise - delayed_gen_noise)[middle]

plt.subplot(2,1,1)
plt.plot(gen_noise)
plt.show()