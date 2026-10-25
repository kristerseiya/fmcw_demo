
import numpy as np
import chirpgen
import matplotlib.pyplot as plt

modulation = "sinusoidal"
Tchirp = 1e-6
Trep = 2e-6
sample_rate = 1e9
B = 3e9
K = 1
phase_noise_stdv = 4
awgn_stdv = 1

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                    Trep=Trep, 
                                    sample_rate=sample_rate,
                                    bandwidth=B,
                                    modulation=modulation,
                                    generation_type="continuous",
                                    lambd_c=1e-3,
                                    dechirp=True,
                                    phase_noise_stdv=B*0.01,
                                    additive_noise_stdv=awgn_stdv,
                                    distance_fall_off=False,
                                    phase_noise_type="stationary_freq_from_psd",
                                    psd_fun=psd_fun,
                                    complex_valued=True,
                                    transmitted_power=1,
                                    )

generator = chirpgen.ChirpGenerator(config)
t = np.arange(0, Trep, 1/sample_rate)
            
delays = np.random.rand(K) * Trep
distance = delays * 3e8 / 2
normalized_delay = delays / Trep

y, (phase_noise_ts, phase_noise) = generator.generate(t, distance, reflection_coef=1, return_phasenoise=True)

deriv_noise = np.diff(phase_noise)/np.diff(phase_noise_ts)
print(np.std(deriv_noise))
plt.subplot(2,1,1)
plt.plot(phase_noise_ts, phase_noise)
plt.subplot(2,1,2)
plt.plot(deriv_noise)
plt.show()