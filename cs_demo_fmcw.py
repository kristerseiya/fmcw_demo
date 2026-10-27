
import numpy as np
import matplotlib.pyplot as plt
import chirpgen
import time_freq
import filters
from filters import SignalModel
import algo.sparse
from complex_sparse_viz import visualize

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['Set2'].colors)
set_colors = plt.colormaps['Set1'].colors

Tchirp = 1e-6
Trep = 2e-6
sample_rate = 1e9

bandwidth = 10e9

def psd_fun(f):
    return (np.abs(f)<0.005*sample_rate)

config = chirpgen.ChirpGeneratorConfig(Tchirp=Tchirp,
                                        Trep=Trep, 
                                        sample_rate=sample_rate,
                                        bandwidth=bandwidth,
                                        modulation="sinusoidal",
                                        generation_type="continuous",
                                        lambd_c=1e-3,
                                        dechirp=True,
                                        phase_noise_stdv=bandwidth*0.005,
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
distance = np.array([14.5, 20, 24])
# refl = [1, 1, 1]
refl = [0.1, 1, 0.01]
y = generator.generate(t, distance, reflection_coef=refl)

def normalize(x):
    return x / np.amax(x)

model = SignalModel(config, model_type="config")
ynew = np.conj(y)*model.get_complex_waveform(t)
M = np.ceil(bandwidth/sample_rate).astype(int)*10
mp_delays, mp_amplitude, mf0 = algo.sparse.matching_pursuit(model, t, ynew, M, refine=True, 
                                                            orthogonalize=False, n_iter=10, 
                                                            get_first_mf_response=True)
mf0 = np.abs(mf0)
mf0 /= np.amax(mf0)
bmp_delays, bmp_amplitude, bmf0 = algo.sparse.bartlett_matching_pursuit(model, t, ynew, M, window_size=100,
                                                                     refine=True, 
                                                                     orthogonalize=False, n_iter=10, 
                                                                     get_first_mf_response=True)
bmf0 = np.abs(bmf0)
bmf0 /= np.amax(bmf0)


bp = algo.sparse.basis_pursuit(model, t, ynew, uprate=M, lambd=10, rho=1000, n_iter=1000, tol=1e-7)
bbp = algo.sparse.bartlett_basis_pursuit(model, t, ynew, uprate=M, lambd=100, 
                                                   window_len=500, stride=100, rho=1000,
                                                   window="blackmanharris", n_iter=1000)
bbp = np.mean(np.abs(bbp), axis=0)
bbp /= np.amax(bbp)
blrbp = algo.sparse.bartlett_low_rank_basis_pursuit(model, t, ynew, uprate=M, lambd=1, 
                                                            window_len=500, stride=100, rho=10,
                                                            group_len=10000, group_stride=5000,
                                                            window="cosine", n_iter=200, tol=0)
blrbp = np.mean(np.abs(blrbp), axis=0)
blrbp /= np.amax(blrbp)

plt.figure(figsize=(16, 4))
plt.subplot(131)
f, t, x = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
plt.pcolormesh(t, f, np.abs(x))
plt.title("STFT")

plt.subplot(132)
stems = plt.stem(mp_delays, np.abs(mp_amplitude), linefmt='--', markerfmt='o', basefmt=' ', label="mp")
plt.plot(np.arange(len(mf0))/M, mf0, label="mp", color=set_colors[0])
plt.setp(stems, color=set_colors[0])

stems = plt.stem(bmp_delays, np.mean(np.abs(bmp_amplitude),axis=1), linefmt='--', markerfmt='o', basefmt=' ', label="bmp")
plt.plot(np.arange(len(bmf0))/M, bmf0, label="bmp", color=set_colors[1])
plt.setp(stems, color=set_colors[1])

plt.plot(np.arange(len(bp))/M,np.abs(bp), zorder=2, label="bp", color=set_colors[1])
plt.plot(np.arange(len(bbp))/M,np.abs(bbp), zorder=3, label="bbp", color=set_colors[2])
plt.plot(np.arange(len(blrbp))/M,np.abs(blrbp), zorder=3, label="bbp-lowrank", color=set_colors[3])
plt.vlines((distance*2/3e8*sample_rate), ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color=set_colors[4])
plt.scatter((distance*2/3e8*sample_rate), np.abs(refl), marker="*", zorder=5, c=[set_colors[4],])
# plt.yscale("log")
# plt.ylim(1e-5, 3)
plt.ylim(0, 1)

plt.subplot(133)
stems = plt.stem(mp_delays, np.abs(mp_amplitude), linefmt='--', markerfmt='o', basefmt=' ', label="mp")
plt.plot(np.arange(len(mf0))/M, mf0, label="mp", color=set_colors[0])
plt.setp(stems, color=set_colors[0])

stems = plt.stem(bmp_delays, np.mean(np.abs(bmp_amplitude),axis=1), linefmt='--', markerfmt='o', basefmt=' ', label="bmp")
plt.plot(np.arange(len(bmf0))/M, bmf0, label="bmp", color=set_colors[1])
plt.setp(stems, color=set_colors[1])

plt.plot(np.arange(len(bp))/M,np.abs(bp), zorder=2, label="bp", color=set_colors[1])
plt.plot(np.arange(len(bbp))/M,np.abs(bbp), zorder=3, label="bbp", color=set_colors[2])
plt.plot(np.arange(len(blrbp))/M,np.abs(blrbp), zorder=3, label="bbp-lowrank", color=set_colors[3])
plt.vlines((distance*2/3e8*sample_rate), ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color=set_colors[4])
plt.scatter((distance*2/3e8*sample_rate), np.abs(refl), marker="*", zorder=5, c=[set_colors[4],])
plt.yscale("log")
plt.ylim(1e-4, 3)
plt.legend()

plt.title("Modulus Amplitude and Time Delay Estimates")

# plt.figure()
# # plt.stem(normalized_delays*len(lasso_amp), np.abs(amplitude), linefmt='g--', markerfmt='o', basefmt=' ', label="omp")
# # plt.plot(np.arange(len(lasso_amp)),np.abs(lasso_amp)/np.amax(np.abs(lasso_amp)), zorder=2, label="bp", color="dodgerblue")
# # plt.plot(np.arange(len(lasso_amp2)),np.abs(lasso_amp2)/np.amax(np.abs(lasso_amp2)), zorder=3, label="bbp", color="orange")
# # plt.plot(np.arange(len(lasso_amp3)),np.abs(lasso_amp3)/np.amax(np.abs(lasso_amp3)), zorder=3, label="bbp-lowrank", color="magenta")
# stems = plt.stem(normalized_delays*len(lasso_amp), np.abs(amplitude), linefmt='--', markerfmt='o', basefmt=' ', label="omp")
# plt.setp(stems, color=set2_colors[0])
# plt.plot(np.arange(len(lasso_amp)),np.abs(lasso_amp)/np.amax(np.abs(lasso_amp)), zorder=2, label="bp", color=set2_colors[1])
# plt.plot(np.arange(len(lasso_amp2)),np.abs(lasso_amp2)/np.amax(np.abs(lasso_amp2)), zorder=3, label="bbp", color=set2_colors[2])
# plt.plot(np.arange(len(lasso_amp3)),np.abs(lasso_amp3)/np.amax(np.abs(lasso_amp3)), zorder=3, label="bbp-lowrank", color=set2_colors[3])
# plt.vlines((distance)*2/3e8/Trep*len(lasso_amp), ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color=set2_colors[4])
# plt.scatter((distance)*2/3e8/Trep*len(lasso_amp), np.abs(refl), marker="*", zorder=5, c=set2_colors[4])
# plt.legend()
plt.show()