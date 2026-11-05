
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time_freq
from scipy.fft import fft, fftfreq

import argparse
from data_run.run import load_data, get_algo_metadata
from algorithm import process_segment

p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--data", default="lnoi_edbr_triangular", help=("either\n"
                                                        "\"lnoi_edbr_triangular\" or \n"
                                                    "\"0821_laser2_triangular\" or \n"
                                                    "\"0821_laser2_sinusoidal\""))
p.add_argument("--frame-num", default=1, type=int)
p.add_argument("--min-distance", default=None, type=float)
p.add_argument("--max-distance", default=None, type=float)
p.add_argument("--load-if", default=None, type=str)
p.add_argument("--inverse", default="ridge", type=str, help="\"ridge\" or \"pinv\" or \"fft\"")
p.add_argument("--if-est", default="hilbert", type=str, help="\"hilbert\" or \"fcrm\"")
p.add_argument("--use-collimator", action="store_true", help="use collimator to estimate transmitted IF")
p.add_argument("--subtract-collimator", action="store_true", help="subtract collimator")
p.add_argument("--downsample", default=1, type=int)
p.add_argument("--start", type=int, default=0, help="first row of x to use")
args = p.parse_args()

args.algo = "resample"
x, meta, _ = load_data(args)
idx1 = meta["L"] * (args.frame_num-1) + args.start
idx2 = meta["L"] * (args.frame_num+2) + args.start
del_t = meta["del_t"]
xi = x[idx1:idx2]
ts_i = xi[:,-1]

algo1_result = process_segment(xi, None, meta, return_intermediate=True)

args.algo = "mf"
args.conjugate = False
args.subtract_collimator = False
args.if_est = "fcrm"
args.use_collimator = True
meta = get_algo_metadata(meta, args)
algo2_result = process_segment(xi, None, meta, return_intermediate=True)

args.algo = "mf"
args.subtract_collimator = True
args.if_est = "fcrm"
args.use_collimator = True
meta = get_algo_metadata(meta, args)
algo3_result = process_segment(xi, None, meta, return_intermediate=True)

args.algo = "mf"
args.subtract_collimator = True
args.if_est = "hilbert"
args.use_collimator = False
args.conjugate = False
meta = get_algo_metadata(meta, args)
algo4_result = process_segment(xi, None, meta, return_intermediate=True)

sig_cbf, sig_cbfnew = algo1_result["before_after_resample"]
ref_if_t3, ref_if_est3 = algo3_result["if_estimate"][:,0], algo3_result["if_estimate"][:,1]
col_if_t3, col_if_est3 = algo3_result["ifcol_estimate"][:,0], algo3_result["ifcol_estimate"][:,1]
ref_if_t4, ref_if_est4 = algo4_result["if_estimate"][:,0], algo4_result["if_estimate"][:,1]

a_est3 = algo3_result["modf_estimate"]
a_est4 = algo4_result["modf_estimate"]

dist_crosscorr2, crosscorr2 = algo2_result["crosscorrelation"][:,0], algo2_result["crosscorrelation"][:,1] 
dist_crosscorr3, crosscorr3 = algo3_result["crosscorrelation"][:,0], algo3_result["crosscorrelation"][:,1] 
dist_crosscorr4, crosscorr4 = algo4_result["crosscorrelation"][:,0], algo4_result["crosscorrelation"][:,1] 
# peak_locs, peak_heights = algo4_result["distances"], algo4_result["peak_strength"]
col_signal3, subtracted_signal3 = algo3_result["subtracted_signal"]


periodogram = np.abs(fft(sig_cbfnew, n=len(sig_cbfnew)*10))
dist_periodogram = fftfreq(len(periodogram)) / del_t * meta["T"] / meta["B"] * 3e8 / 4
shared_region = (dist_periodogram>=dist_crosscorr2[0])*(dist_periodogram<=dist_crosscorr2[-1])

crosscorr2 /= np.amax(crosscorr2)
crosscorr3 /= np.amax(crosscorr3)
crosscorr4 /= np.amax(crosscorr4)
periodogram = periodogram / np.amax(periodogram[shared_region])

plt.figure()
plt.subplot(1,2,1)
stft_f, stft_t, stft_x = time_freq.short_time_fourier_transform(xi[:,0], 64, 16, 1/del_t, zero_pad=192)
plt.pcolormesh(stft_t, stft_f, np.log10(np.abs(stft_x)))
plt.title("STFT of scene signal")
plt.subplot(1,2,2)
stft_f, stft_t, stft_x = time_freq.short_time_fourier_transform(xi[:,1], 64, 16, 1/del_t, zero_pad=192)
plt.pcolormesh(stft_t, stft_f, np.log10(np.abs(stft_x)))
plt.title("STFT of reference")
plt.tight_layout()


plt.figure(figsize=(14,7))
plt.subplot(2,3,1)
stft_f, stft_t, stft_x = time_freq.short_time_fourier_transform(sig_cbfnew, 64, 16, 1/del_t, zero_pad=192)
plt.pcolormesh(stft_t, stft_f, np.log10(np.abs(stft_x)))
plt.title("STFT of resampled scene signal")
plt.subplot(2,3,2)
stft_f, stft_t, stft_x = time_freq.short_time_fourier_transform(col_signal3, 64, 16, 1/del_t, zero_pad=192)
plt.pcolormesh(stft_t, stft_f, np.log10(np.abs(stft_x)))
plt.title("STFT of subtracted scene signal")
plt.subplot(2,3,3)
stft_f, stft_t, stft_x = time_freq.short_time_fourier_transform(subtracted_signal3, 64, 16, 1/del_t, zero_pad=192)
plt.pcolormesh(stft_t, stft_f, np.log10(np.abs(stft_x)))
plt.title("STFT of subtracted scene signal")
plt.subplot(2,3,4)
plt.plot(ref_if_t3, ref_if_est3, label="ref (fcrm)", color="dodgerblue")
plt.plot(col_if_t3, col_if_est3, label="col (fcrm)", color="orange")
plt.plot(ref_if_t4, ref_if_est4, label="ref (hilbert)", color="red")
plt.title("Inst freq estimate of reference")
plt.legend()
plt.subplot(2,3,5)
plt.plot(ts_i-ts_i[0], a_est3-np.amin(a_est3), label="double delay", color="dodgerblue")
plt.plot(ts_i-ts_i[0], a_est4-np.amin(a_est4), label="single delay", color="red")
plt.title("(transmitted) Inst freq estimate")
plt.legend()
plt.subplot(2,3,6)
plt.plot(dist_crosscorr2, crosscorr2, label="double delay without collimator subtraction", color="green")
plt.plot(dist_crosscorr3, crosscorr3, label="double delay with collimator subtraction", color="dodgerblue")
plt.plot(dist_crosscorr4, crosscorr4, label="single delay with collimator subtraction", color="red")
# plt.scatter(peak_locs, peak_heights, marker="x", color="orange")
plt.plot(dist_periodogram, periodogram, label="resampling", color="purple")
plt.title("MF response vs resampled FT")
plt.xlim(dist_crosscorr2[0], dist_crosscorr2[-1])
plt.legend()
plt.ylim(0, 1.5)

plt.tight_layout()

plt.show()
