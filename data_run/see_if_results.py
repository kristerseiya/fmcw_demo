
import numpy as np
import matplotlib.pyplot as plt
from fmcw_aux import ModuFuncSolver

def _mask_convolve(x1, x2, mask, max_shift):
    assert(len(x1)==len(x2))
    x1_pad = np.pad(x1, (max_shift, max_shift), mode="constant", constant_values=0)

    shifts = np.arange(-max_shift, max_shift+1)
    convolved = np.zeros((len(shifts),))
    for i in shifts:
        convolved[i+max_shift] = np.sum( x1_pad[i+max_shift:i+max_shift+len(x2)] * x2 * mask )
    return convolved


results = np.load("data_run/results/sin_if.npz")
T = results["config_T"]
print(results["config_ref_if_smooth_length"])
del_t = results["config_del_t"]
L = results["config_L"]
if_est = results["ref_if"]
ift1, ifq1 = if_est[0,:,0], if_est[0,:,1]
mask = 1*(ift1>=T)*(ift1<2*T)
max_shifts = 300
skip = 1000
best_shifts = np.zeros(((len(if_est)+skip-1)//skip, (len(if_est)+skip-1)//skip))
# for i in range(0, len(if_est)):
avg_slope = 0
cnt = 0
for i in range(0, len(if_est), skip):
    ifti, ifqi = if_est[i,:,0], if_est[i,:,1]
    print(i)
    for j in range(i+skip, len(if_est), skip):
        iftj, ifqj = if_est[j,:,0], if_est[j,:,1]
        convolved = _mask_convolve(ifqi, ifqj, mask, max_shifts)
        best_shift = np.argmax(convolved) - max_shifts
        best_shifts[i//skip,j//skip] = best_shift
        avg_slope += best_shift/(j-i)
        cnt += 1

avg_slope /= cnt
print(avg_slope)

avg_ifq = np.zeros_like(ifq1)
naive_avg_ifq = np.zeros_like(ifq1)
for i in range(0, len(if_est), 1):
    ifti, ifqi = if_est[i,:,0], if_est[i,:,1]
    naive_avg_ifq += ifqi
    if i > 0:
        ifqi = np.interp(ifti, ifti+avg_slope*i*del_t, ifqi, period=3*T)
    avg_ifq += ifqi
avg_ifq /= len(if_est)
naive_avg_ifq /= len(if_est)

# results["config_modfsolver"
t = np.arange(len(if_est))

modufuncsolver = ModuFuncSolver(L*3, 1/del_t, 
                                [results["config_aux_delay"],], period=3*L*del_t, 
                                method=results["config_inv"], verbose=True)
a_est = results["modf"]
naive_avg_a_est = modufuncsolver.estimate((ift1, naive_avg_ifq*del_t))/del_t
avg_a_est = modufuncsolver.estimate((ift1, avg_ifq*del_t))/del_t
plt.subplot(221)
plt.plot(np.arange(0, len(if_est), skip), best_shifts[0])
plt.plot(t*avg_slope)
plt.subplot(222)
plt.pcolormesh(best_shifts)
plt.subplot(223)
plt.plot(ifq1, label="example 1")
plt.plot(if_est[3000,:,1], label="example 2")
plt.plot(if_est[6000,:,1], label="example 3")
plt.plot(naive_avg_ifq, label="naive average")
plt.plot(avg_ifq, label="drift adjusted average")
plt.subplot(224)
plt.plot(a_est[0,:], label="example 1")
plt.plot(a_est[3000,:], label="example 2")
plt.plot(a_est[6000,:], label="example 3")
plt.plot(naive_avg_a_est, label="naive average")
plt.plot(avg_a_est, label="drift adjusted average")
plt.legend()
plt.show()