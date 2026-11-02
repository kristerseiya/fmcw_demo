
import numpy as np
import matplotlib.pyplot as plt
import chirpgen
import time_freq
import filters
from filters import SignalModel
import algo.sparse
from complex_sparse_viz import visualize
import fmcw_aux
import algo.filter

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['Set1'].colors)
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
                                        phase_noise_stdv=bandwidth*0.03,
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
# distance = np.array([14.51, 20.08, 24.3])
distance = np.array([14.51, 21.59, 24.36])
aux_distance = 21.59
refl = [0.1, 1, 0.01]
y = generator.generate(t, distance, reflection_coef=refl) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.1
aux = generator.generate(t, aux_distance, reflection_coef=1, same_phasenoise=True) + ( np.random.randn(len(t)) + 1j*np.random.randn(len(t)) ) * 0.1
y = np.real(y)
aux = np.real(aux)

d_low, d_high = 12, 30
model0 = SignalModel(config, model_type="config")
ynew = y*model0.get_complex_waveform(t)*(t>=Trep)*(t<Trep*2)
print(len(ynew))
M = np.ceil(bandwidth/sample_rate).astype(int)
print("running bp without auxiliary...")
bp0 = algo.sparse.basis_pursuit_faster(model0, t, ynew, uprate=M, 
                                      lambd=30, rho=1000, n_iter=1000, tol=3e-7,
                                      t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bp0 = _normalize(np.abs(bp0))

bbp0 = algo.sparse.bartlett_basis_pursuit_faster(model0, t, ynew, uprate=M, lambd=30, 
                                                window_len=1000, stride=500, rho=1000,
                                                window="blackmanharris", n_iter=1000, tol=3e-7,
                                                t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bbp0 = _normalize(np.mean(np.abs(bbp0), axis=0))

ift, ifq = fmcw_aux.estimate_if_hilbert(aux, fs=1000, filt=dict(type="fir", fc=20))
ift *= 1/sample_rate
ifq *= sample_rate

modufuncsolver = fmcw_aux.ModuFuncSolver(len(t), sample_rate, 
                                         [aux_distance*2/3e8,], period=3*Trep, 
                                         method="ridge", verbose=True)

a_est = modufuncsolver.estimate((ift, ifq/sample_rate))
a_est *= sample_rate
a_est = np.convolve(a_est, np.ones((101,))/101, mode="same")

model = SignalModel((t, a_est), model_type="inst_freq", period=3*Trep)
aux_model_tx = model.get_complex_waveform(t)
aux_model_rx = model.get_complex_waveform(t-aux_distance*2/3e8)
aux_model = aux_model_tx*np.conj(aux_model_rx)

print("running bp with auxiliary...")
ynew = y*model.get_complex_waveform(t)*(t>=Trep)*(t<Trep*2)
bp = algo.sparse.basis_pursuit_faster(model, t, ynew, uprate=M, 
                                      lambd=30, rho=1000, n_iter=1000, tol=3e-7,
                                      t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bp = _normalize(np.abs(bp))

bbp = algo.sparse.bartlett_basis_pursuit_faster(model, t, ynew, uprate=M, lambd=30, 
                                                window_len=1000, stride=500, rho=1000,
                                                window="blackmanharris", n_iter=1000, tol=3e-7,
                                                t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bbp = _normalize(np.mean(np.abs(bbp), axis=0))

print("running mp with auxiliary...")
mp_delays, mp_amplitude = algo.sparse.matching_pursuit(model, t, ynew, M, refine=True, 
                                                       orthogonalize=False, n_iter=100,
                                                       t_low=d_low*2/3e8, t_high=d_high*2/3e8)
mp_amplitude = _normalize(np.abs(mp_amplitude))

bmp_delays, bmp_amplitude = algo.sparse.bartlett_matching_pursuit(model, t, ynew, M, window_len=300,
                                                                  refine=True, 
                                                                  orthogonalize=False, n_iter=100, 
                                                                  t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bmp_amplitude = _normalize(np.mean(np.abs(bmp_amplitude),axis=1))

print("running mf with auxiliary...")
d_weaklow, d_weakhigh = 23, 26
(mf_delays, mf_peaks), (tmf, mf) = algo.filter.matchedfilter(model, t, ynew, uprate=M,
                                                      t_low=d_weaklow*2/3e8, t_high=d_weakhigh*2/3e8, top_k=10,
                                                      distance=10)
mf = _normalize(mf)

(bmf_delays, bmf_peaks), (tbmf, bmf) = algo.filter.bartlett_matchedfilter(model, t, ynew, uprate=M,
                                                                 window_len=1000, stride=500,
                                                                 window="blackmanharris",
                                                                 t_low=d_weaklow*2/3e8, t_high=d_weakhigh*2/3e8, 
                                                                 top_k=10,
                                                                 distance=10)
bmf = _normalize(bmf)

plt.figure(figsize=(16,12))
plt.subplot(331)
ff, tt, xx = time_freq.short_time_fourier_transform(aux, 32, 8, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.log10(np.abs(xx)))
plt.title("auxiliary signal")
plt.subplot(332)
plt.plot(ift, ifq)
plt.title("IF estimate")
plt.subplot(333)
plt.plot(t, a_est)
plt.title("MF estimate")
plt.subplot(334)
ff, tt, xx = time_freq.short_time_fourier_transform(aux_model, 32, 8, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.log10(np.abs(xx)))
plt.title("auxiliary model")
plt.subplot(335)
ff, tt, xx = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.log10(np.abs(xx)))
plt.title("scene signal")
plt.subplot(336)
plt.plot(np.arange(len(bp0))/M/sample_rate/2*3e8, bp0, zorder=2, label="bp", color="dodgerblue")
plt.plot(np.arange(len(bbp0))/M/sample_rate/2*3e8, bbp0, zorder=3, label="bbp", color="green")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Basis Pursuit without auxiliary")
plt.subplot(337)
plt.plot(tmf/2*3e8, mf, zorder=2, label="mf", color="purple")
plt.plot(tbmf/2*3e8, bmf, zorder=3, label="bmf", color="orange")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_weaklow, d_weakhigh)
plt.legend()
plt.title("Matched Filter with auxiliary")
plt.subplot(338)
plt.scatter(mp_delays/sample_rate/2*3e8, mp_amplitude, marker="o", c="purple", label="mp")
plt.scatter(bmp_delays/sample_rate/2*3e8, bmp_amplitude, marker="o", c="orange", label="bmp")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Matching Pursuit with auxiliary")
plt.subplot(339)
plt.plot(np.arange(len(bp))/M/sample_rate/2*3e8, bp, zorder=2, label="bp", color="dodgerblue")
plt.plot(np.arange(len(bbp))/M/sample_rate/2*3e8, bbp, zorder=3, label="bbp", color="green")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Basis Pursuit with auxiliary")
plt.tight_layout()

print("subtracting strong reflections")
strong_ift, strong_ifq = fmcw_aux.estimate_if_fcrm(y)
strong_ift *= 1/sample_rate
strong_ifq *= sample_rate
y_subtracted = fmcw_aux.subtract_signal(t, y, strong_ift, strong_ifq, 200, period=3*Trep)

ynew = y_subtracted*model0.get_complex_waveform(t)*(t>=Trep)*(t<Trep*2)
print("running bp without auxiliary...")
bp0 = algo.sparse.basis_pursuit_faster(model0, t, ynew, uprate=M, 
                                      lambd=30, rho=1000, n_iter=1000, tol=3e-7,
                                      t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bp0 = _normalize(np.abs(bp0))

bbp0 = algo.sparse.bartlett_basis_pursuit_faster(model0, t, ynew, uprate=M, lambd=30, 
                                                window_len=1000, stride=500, rho=1000,
                                                window="blackmanharris", n_iter=1000, tol=3e-7,
                                                t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bbp0 = _normalize(np.mean(np.abs(bbp0), axis=0))

print("running bp with auxiliary...")
ynew = y_subtracted*model.get_complex_waveform(t)*(t>=Trep)*(t<Trep*2)
bp = algo.sparse.basis_pursuit_faster(model, t, ynew, uprate=M, 
                                      lambd=60, rho=1000, n_iter=1000, tol=3e-7,
                                      t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bp = _normalize(np.abs(bp))

bbp = algo.sparse.bartlett_basis_pursuit_faster(model, t, ynew, uprate=M, lambd=30, 
                                                window_len=1000, stride=500, rho=1000,
                                                window="blackmanharris", n_iter=1000, tol=3e-7,
                                                t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bbp = _normalize(np.mean(np.abs(bbp), axis=0))

print("running mp with auxiliary...")
mp_delays, mp_amplitude = algo.sparse.matching_pursuit(model, t, ynew, M, refine=True, 
                                                       orthogonalize=False, n_iter=100,
                                                       t_low=d_low*2/3e8, t_high=d_high*2/3e8)
mp_amplitude = _normalize(np.abs(mp_amplitude))

bmp_delays, bmp_amplitude = algo.sparse.bartlett_matching_pursuit(model, t, ynew, M, window_len=300,
                                                                  refine=True, 
                                                                  orthogonalize=False, n_iter=100, 
                                                                  t_low=d_low*2/3e8, t_high=d_high*2/3e8)
bmp_amplitude = _normalize(np.mean(np.abs(bmp_amplitude),axis=1))

print("running mf with auxiliary...")
d_weaklow, d_weakhigh = 23, 26
(mf_delays, mf_peaks), (tmf, mf) = algo.filter.matchedfilter(model, t, ynew, uprate=M,
                                                      t_low=d_weaklow*2/3e8, t_high=d_weakhigh*2/3e8, top_k=10,
                                                      distance=10)
mf = _normalize(mf)

(bmf_delays, bmf_peaks), (tbmf, bmf) = algo.filter.bartlett_matchedfilter(model, t, ynew, uprate=M,
                                                                 window_len=1000, stride=500,
                                                                 window="blackmanharris",
                                                                 t_low=d_weaklow*2/3e8, t_high=d_weakhigh*2/3e8, 
                                                                 top_k=10,
                                                                 distance=10)
bmf = _normalize(bmf)

plt.figure(figsize=(16,8))
plt.subplot(231)
plt.plot(strong_ift, strong_ifq)
plt.title("IF estimate of strong reflections")
plt.subplot(232)
ff, tt, xx = time_freq.short_time_fourier_transform(y_subtracted, 32, 8, zero_pad=512-32)
plt.pcolormesh(tt, ff, np.log10(np.abs(xx)))
plt.title("subtracted scene")
plt.subplot(233)
plt.plot(np.arange(len(bp0))/M/sample_rate/2*3e8, bp0, zorder=2, label="bp", color="dodgerblue")
plt.plot(np.arange(len(bbp0))/M/sample_rate/2*3e8, bbp0, zorder=3, label="bbp", color="green")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Basis Pursuit without auxiliary")
plt.subplot(234)
plt.plot(tmf/2*3e8, mf, zorder=2, label="mf", color="purple")
plt.plot(tbmf/2*3e8, bmf, zorder=3, label="bmf", color="orange")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_weaklow, d_weakhigh)
plt.legend()
plt.title("Matched Filter with auxiliary")
plt.subplot(235)
plt.scatter(mp_delays/sample_rate/2*3e8, mp_amplitude, marker="o", c="purple", label="mp")
plt.scatter(bmp_delays/sample_rate/2*3e8, bmp_amplitude, marker="o", c="orange", label="bmp")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Matching Pursuit with auxiliary")
plt.subplot(236)
plt.plot(np.arange(len(bp))/M/sample_rate/2*3e8, bp, zorder=2, label="bp", color="dodgerblue")
plt.plot(np.arange(len(bbp))/M/sample_rate/2*3e8, bbp, zorder=3, label="bbp", color="green")
plt.vlines(distance, ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color="red")
plt.scatter(distance, np.abs(refl), marker="*", zorder=5, c="red")
plt.xlim(d_low, d_high)
plt.legend()
plt.title("Basis Pursuit with auxiliary")
plt.tight_layout()
# model = SignalModel(config, model_type="config")


# ynew = np.conj(y)*model.get_complex_waveform(t)
# M = np.ceil(bandwidth/sample_rate).astype(int)*10
# mp_delays, mp_amplitude, mf0 = algo.sparse.matching_pursuit(model, t, ynew, M, refine=True, 
#                                                             orthogonalize=False, n_iter=10, 
#                                                             get_first_mf_response=True)
# mf0 = np.abs(mf0)
# mf0 /= np.amax(mf0)
# bmp_delays, bmp_amplitude, bmf0 = algo.sparse.bartlett_matching_pursuit(model, t, ynew, M, window_size=300,
#                                                                      refine=True, 
#                                                                      orthogonalize=False, n_iter=10, 
#                                                                      get_first_mf_response=True)
# bmf0 = np.abs(bmf0)
# bmf0 /= np.amax(bmf0)


# bp = algo.sparse.basis_pursuit(model, t, ynew, uprate=M, lambd=150, rho=1000, n_iter=1000, tol=1e-7)
# bp /= np.amax(bp)
# bbp = algo.sparse.bartlett_basis_pursuit(model, t, ynew, uprate=M, lambd=10, 
#                                                    window_len=300, stride=50, rho=1000,
#                                                    window="blackmanharris", n_iter=1000, tol=1e-7)
# bbp = np.mean(np.abs(bbp), axis=0)
# bbp /= np.amax(bbp)
# blrbp = algo.sparse.bartlett_low_rank_basis_pursuit(model, t, ynew, uprate=M, lambd=100, 
#                                                             window_len=300, stride=50, rho=1000,
#                                                             group_len=1000, group_stride=250,
#                                                             window="blackmanharris", n_iter=200, tol=1e-7)
# blrbp = np.mean(np.abs(blrbp), axis=0)
# blrbp /= np.amax(blrbp)

# plt.figure(figsize=(16, 4))
# plt.subplots_adjust(left=0.05, right=0.95)
# plt.subplot(131)
# f, t, x = time_freq.short_time_fourier_transform(y, 32, 8, zero_pad=512-32)
# plt.pcolormesh(t, f, np.abs(x))
# plt.title("STFT")

# plt.subplot(132)
# stems = plt.stem(mp_delays, np.abs(mp_amplitude), linefmt='--', markerfmt='o', basefmt=' ', label="mp")
# plt.plot(np.arange(len(mf0))/M, mf0, label="mf", color=set_colors[0], alpha=0.5)
# plt.setp(stems, color=set_colors[0])

# stems = plt.stem(bmp_delays, np.mean(np.abs(bmp_amplitude),axis=1), linefmt='--', markerfmt='o', basefmt=' ', label="bmp")
# plt.plot(np.arange(len(bmf0))/M, bmf0, label="bmf", color=set_colors[1], alpha=0.5)
# plt.setp(stems, color=set_colors[1])

# plt.plot(np.arange(len(bp))/M,np.abs(bp), zorder=2, label="bp", color=set_colors[6])
# plt.plot(np.arange(len(bbp))/M,np.abs(bbp), zorder=3, label="bbp", color=set_colors[2])
# plt.plot(np.arange(len(blrbp))/M,np.abs(blrbp), zorder=3, label="bbp-lowrank", color=set_colors[3])
# plt.vlines((distance*2/3e8*sample_rate), ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color=set_colors[4])
# plt.scatter((distance*2/3e8*sample_rate), np.abs(refl), marker="*", zorder=5, c=[set_colors[4],])
# # plt.yscale("log")
# # plt.ylim(1e-5, 3)
# plt.ylim(0, 1)
# plt.legend(loc="upper right")

# plt.subplot(133)
# stems = plt.stem(mp_delays, np.abs(mp_amplitude), linefmt='--', markerfmt='o', basefmt=' ', label="mp")
# plt.plot(np.arange(len(mf0))/M, mf0, label="mf", color=set_colors[0], alpha=0.5)
# plt.setp(stems, color=set_colors[0])

# stems = plt.stem(bmp_delays, np.mean(np.abs(bmp_amplitude),axis=1), linefmt='--', markerfmt='o', basefmt=' ', label="bmp")
# plt.plot(np.arange(len(bmf0))/M, bmf0, label="bmf", color=set_colors[1], alpha=0.5)
# plt.setp(stems, color=set_colors[1])

# plt.plot(np.arange(len(bp))/M,np.abs(bp), zorder=2, label="bp", color=set_colors[6])
# plt.plot(np.arange(len(bbp))/M,np.abs(bbp), zorder=3, label="bbp", color=set_colors[2])
# plt.plot(np.arange(len(blrbp))/M,np.abs(blrbp), zorder=3, label="bbp-lowrank", color=set_colors[3])
# plt.vlines((distance*2/3e8*sample_rate), ymin=0, ymax=1, linestyle="-", zorder=1, label="true", color=set_colors[4])
# plt.scatter((distance*2/3e8*sample_rate), np.abs(refl), marker="*", zorder=5, c=[set_colors[4],])
# plt.yscale("log")
# plt.ylim(1e-4, 3)
# plt.legend(loc="upper right")

# plt.title("Modulus Amplitude and Time Delay Estimates")

plt.show()