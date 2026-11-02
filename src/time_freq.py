
import numpy as np
from scipy import signal, linalg, ndimage
import scipy.fftpack


def wigner_distribution(t, x, use_analytic=False, sample_frequency=None,
                        t_0=0, t_1=1, flip_frequency_range=True):
    """Discrete Pseudo Wigner Ville Distribution based on [1]
    Args:
        x, array like, signal input array of length N
        use_analytic, bool, whether or not to use analytic associate of input
            data x by default set to True
        sample_frequency, sampling frequency
        t_0, time at which the first sample was recorded
        t_1, time at which the last sample was recorded
        flip_frequency_range, flip the data in about the time axis such that
            the minimum frequency is in the left bottom corner.
    Returns:
        wigner_distribution, N x N matrix
        frequency_bins, array like, length N frequency range
    References:
        [1] T. Claasen & W. Mecklenbraeuker, The Wigner Distribution -- A Tool
        For Time-Frequency Signal Analysis, Phillips J. Res. 35, 276-300, 1980
    """

    # Ensure the input array is a np array
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    # Compute the autocorrelation function matrix
    if x.ndim != 1:
        raise ValueError("Input data should be one dimensional time series.")
    # Use analytic associate if set to True
    if use_analytic:
        if all(np.isreal(x)):
            x = signal.hilbert(x)
        else:
            raise RuntimeError("Keyword 'use_analytic' set to True but signal"
                               " is of complex data type, but analytic signals"
                               " must be real valued")

    # calculate the wigner distribution
    N = x.shape[0]
    bins = np.arange(N)
    indices = linalg.hankel(bins, bins + N - (N % 2))

    # padded_x = np.pad(x, (N, N), 'constant', constant_values=0)
    # wigner_integrand = padded_x[indices+N] * np.conjugate(padded_x[indices[::, ::-1]])
    padded_x = np.pad(x, (N, N), 'constant', constant_values=0)
    wigner_integrand = padded_x[indices+N] * np.conjugate(padded_x[indices[::, ::-1]])

    wigner_distribution = np.real(np.fft.fft(wigner_integrand, axis=1)).T

    # calculate sample frequency
    if sample_frequency is None:
        sample_frequency = N / (t_1 - t_0)

    # calculate frequency range
    if use_analytic:
        max_frequency = sample_frequency/2
    else:
        max_frequency = sample_frequency/4

    # flip the frequency range
    if flip_frequency_range:
        wigner_distribution = wigner_distribution[::-1, ::]

    return wigner_distribution, max_frequency
    # if use_analytic:
    #     if all(np.isreal(x)):
    #         x = signal.hilbert(x)
    #     else:
    #         raise RuntimeError("Keyword 'use_analytic' set to True but signal"
    #                            " is of complex data type, but analytic signals"
    #                            " must be real valued")

    # N = len(x)

    # if (N <= 0 ): raise ValueError("Number of Frequency bins N must be greater than zero.")

    # tfr = np.zeros([N, N], dtype='complex')
    # for icol in np.arange(N):
    #     #ti = t[icol]
    #     ti = icol
    #     taumax = min([ti, N-ti-1, int(round(N/2.0))-1])
    #     tau = np.arange(-taumax, taumax+1)
    #     indices = ((N+tau)%N)
    #     tfr[np.ix_(indices, [icol])] = np.transpose(np.array(x[ti+tau] * np.conj(x[ti-tau]), ndmin=2))
    #     tau=int(round(N/2))+1
    #     if ((ti+1) <= (N-tau)) and ((ti+1) >= (tau+1)):
    #         if(tau >= tfr.shape[0]): tfr = np.append(tfr, np.zeros([1, N]), axis=0)
    #         tfr[np.ix_([tau], [icol])] = np.array(0.5 * (x[ti+tau] * np.conj(x[ti-tau]) + x[ti-tau] * np.conj(x[ti+tau])))

    # tfr = np.real(np.fft.fft(tfr, axis=0))
    # f = 0.5*np.arange(N)/float(N)
    # return (tfr), t


def interference_reduced_wigner_distribution(
        wigner_distribution, number_smoothing_steps=16,
        t_filt_max_percentage=0.03, f_filt_max_percentage=0.02):
    """Method for reducing interference terms based on [1]
    Params:
        wigner_distribution, array like, N x N discrete wigner distribution
        matrix
    Returns:
        interference reduced wigner distribution, N x N ndarray
    Uses a method for interference reduction based on Pikula et al. [1].
    The method works by executing multiple smoothings using a gaussian
    filter, in this implementation using the scipy.ndimage module. The
    optimal smoothing per time-frequency bin is then chosen. Pikula et al.
    [1] goes into more detail on how this optimal smoothing can be chosen.
    The output is then a distribution which contains mainly autoterms with
    strongly suppressed interference terms, better representing the actual
    signal that is present. This, however, destroys many of the
    distributions' mathematical properties, and should only serve as an
    analysis tool for autoterms.
    References:
        [1] Pikula, Stanislav & Beneš, Petr. (2020). A New Method for
        Interference Reduction in the Smoothed Pseudo Wigner-Ville
        Distribution. International Journal on Smart Sensing and
        Intelligent Systems. 7. 1-5. 10.21307/ijssis-2019-101.
    """
    # Ensure the input array is a np array
    if not isinstance(wigner_distribution, np.ndarray):
        wigner_distribution = np.asarray(wigner_distribution)
    # Compute the autocorrelation function matrix
    if wigner_distribution.ndim != 2:
        raise ValueError("Input data should be a two dimensional discrete"
                         " wigner distribution.")
    N_f, N_t = wigner_distribution.shape
    t_filter_widths = np.linspace(0, N_t * t_filt_max_percentage, number_smoothing_steps)
    f_filter_widths = np.linspace(0, N_f * f_filt_max_percentage, number_smoothing_steps)

    # filter at various filtration widths
    smoothed_wigner_distributions = np.zeros((number_smoothing_steps, N_f, N_t))
    for i, (f_fw, t_fw) in enumerate(zip(t_filter_widths, f_filter_widths)):
        smoothed_wigner_distributions[i] = ndimage.gaussian_filter(wigner_distribution, sigma=(f_fw, t_fw))

    # differential analysis per time-frequency bin
    first_derivative = np.diff(smoothed_wigner_distributions, axis=0)
    smoothing_index_best_guess = np.argmax(np.abs(first_derivative),
                                              axis=0)

    # choose smoothing per time-frequency bin
    x, y, z = smoothed_wigner_distributions.shape
    interference_reduced_wigner_distribution = smoothed_wigner_distributions[smoothing_index_best_guess,
                                                                             np.arange(y)[::, np.newaxis],
                                                                             np.arange(z)[np.newaxis, ::]]
    return interference_reduced_wigner_distribution

def short_time_fourier_transform(x, window, stride=1, sample_rate=1, real=False, t0=0, zero_pad=64):
    
    N = len(x)
    if isinstance(window, int):
        window_size = window
        t = np.arange(-window_size//2, window_size-window_size//2)
        h = np.cos(np.pi*t/window_size)**2
    elif isinstance(window, np.ndarray):
        window_size = len(window)
        h = window
    else:
        raise ValueError("window must be either int or 1d numpy array")
    
    start_idx = np.arange(0,N+1-window_size,stride)
    tf_plot = np.zeros((window_size+zero_pad,len(start_idx)), dtype=np.complex128)
    for j, i in enumerate(start_idx):
        y = x[i:(i+window_size)] * h
        if zero_pad > 0:
            y = np.pad(y, (0, zero_pad), mode="constant", constant_values=0)
        tf_plot[:,j] =  scipy.fftpack.fft(y)
    negfreq = tf_plot[(window_size+zero_pad+1)//2:, :]
    posfreq = tf_plot[:(window_size+zero_pad+1)//2, :]
    if real:
        tf_plot = posfreq
    else:
        tf_plot = np.concatenate([negfreq, posfreq], axis=0)

    if real:
        stft_f = np.arange(tf_plot.shape[0]) / tf_plot.shape[0] * sample_rate / 2
    else:
        stft_f = np.arange(tf_plot.shape[0]) / tf_plot.shape[0] * sample_rate
        stft_f = stft_f - sample_rate/2
    stft_t = np.arange(tf_plot.shape[1]) * stride/sample_rate + window_size/sample_rate/2 + t0
    return stft_f, stft_t, tf_plot

def reassignment_method(x, window_length, hop_length, thres=0):
    t = np.arange(-window_length//2, window_length-window_length//2)
    h = np.cos(np.pi*t/window_length)**2
    # dh = 2*(-np.sin(np.pi*t/window_length))*np.pi/window_length*h
    dh = 2*np.cos(np.pi*t/window_length)*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    # f = np.arange(0, 0.5-0.5/len(h), 1/len(h))
    f, t, sh = short_time_fourier_transform(x, h, hop_length, zero_pad=0)#[:(len(h)-len(h)//2),:]
    _, _, sdh = short_time_fourier_transform(x, dh, hop_length, zero_pad=0)#[:(len(h)-len(h)//2),:]

    ff = np.stack([f]*sh.shape[1], axis=1)
    valid = np.abs(sh) > thres
    newf = ff[valid]-np.imag(sdh[valid]/sh[valid]/2/np.pi)
    t = np.arange(sh.shape[1]).astype(float)
    t = np.stack([t]*sh.shape[0], axis=0)
    t[~valid] = np.nan
    f = np.zeros(sh.shape)
    f[:] = np.nan
    f[valid] = newf
    mags = np.abs(sh)
    mags[~valid] = np.nan
    return t, f, mags

def second_order_reassignment_method(x, window_length, hop_length, thres=0):
    t = np.arange(-window_length//2, window_length-window_length//2)
    h = np.cos(np.pi*t/window_length)**2
    dh = 2*np.cos(np.pi*t/window_length)*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    th = t * h
    ddh = 2*(-np.sin(np.pi*t/window_length))*np.pi/window_length*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    ddh = ddh + 2*np.cos(np.pi*t/window_length)*(-np.cos(np.pi*t/window_length))*np.pi/window_length*np.pi/window_length
    tdh = t * dh
    f, t, sh = short_time_fourier_transform(x, h, hop_length)
    _, _, sdh = short_time_fourier_transform(x, dh, hop_length)
    _, _, sth = short_time_fourier_transform(x, th, hop_length)
    _, _, sddh = short_time_fourier_transform(x, ddh, hop_length)
    _, _, stdh = short_time_fourier_transform(x, tdh, hop_length)

    ff = np.stack([f]*sh.shape[1], axis=1)
    tt = np.arange(sh.shape[1]).astype(float)
    tt = np.stack([tt]*sh.shape[0], axis=0)
    valid1 = np.abs(sh) > thres
    valid2 = valid1 * (np.abs(sth*sdh-stdh*sh) > thres)
    f_tilde_ = ff[valid1] - sdh[valid1]/sh[valid1]/(1j*2*np.pi)
    f_tilde = ff[valid2] - sdh[valid2]/sh[valid2]/(1j*2*np.pi)
    t_tilde = tt[valid2] + sth[valid2]/sh[valid2]
    q_tilde = (sddh[valid2]*sh[valid2]-(sdh[valid2])**2) / (sth[valid2]*sdh[valid2]-stdh[valid2]*sh[valid2]) / (1j*2*np.pi)
    
    tt[~valid1] = np.nan
    f = np.zeros(sh.shape)
    f[:] = np.nan
    f[valid1] = np.real(f_tilde_)
    f[valid2] = np.real(f_tilde + q_tilde * (tt[valid2] - t_tilde))
    mags = np.abs(sh)
    mags[~valid1] = np.nan
    return tt, f, mags

def ridge_detection(times, freqs, mags, gamma, c1, c2, n_survivor=7):
    # mags = np.copy(mags)
    # mags[np.isnan(mags)] = 0
    # times = np.copy(times)
    # times[np.isnan(times)] = 0
    # freqs = np.copy(freqs)
    # freqs[np.isnan(freqs)] = 0
    max_idx_col = np.argmax(mags, 1)
    survivors = np.zeros((n_survivor, mags.shape[1]), dtype=int)
    path_dist = np.zeros((n_survivor,))

    # first
    valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,0])]
    if (len(valididx)<n_survivor):
            print(len(valididx))
    dist_metric = - 20*np.log10(mags[valididx,0]+1e-10)
    bestidx = np.argpartition(dist_metric, n_survivor)[:n_survivor]
    survivors[:,0] = valididx[bestidx]
    path_dist = dist_metric[bestidx]

    # second
    valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,1])]
    if (len(valididx)<n_survivor):
            print(len(valididx))
    diff1 = - np.reshape(freqs[survivors[:,0],0], (-1,1)) + freqs[valididx,1]
    diff1 /= (times[valididx[0],1]-times[valididx[0],0])
    dist_metric = np.reshape(path_dist, (-1,1)) - 20*np.log10(mags[valididx,1]+1e-10) + np.minimum(np.abs(diff1), np.abs(diff1-gamma))*c1
    bestidx = np.argpartition(dist_metric.flatten(), n_survivor)[:n_survivor]
    # print(dist_metric)
    # print(dist_metric.flatten()[bestidx])
    
    bestidx = np.unravel_index(bestidx, dist_metric.shape)
    # print(dist_metric[bestidx[0], bestidx[1]])
    # exit()
    new_survivors = np.zeros((n_survivor, 2))
    for n in range(n_survivor):
        new_survivors[n,:1] = survivors[bestidx[0][n],:1]
        new_survivors[n,-1] = valididx[bestidx[1][n]]
        path_dist[n] = dist_metric[bestidx[0][n], bestidx[1][n]]
    survivors[:,:2] = new_survivors

    # rest
    for i in range(2, mags.shape[1]):

        valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,i])]
        if (len(valididx)<n_survivor):
            print(len(valididx))
        diff1 = - np.reshape(freqs[survivors[:,i-1],i-1], (-1,1)) + freqs[valididx,i]
        diff1 /= (times[valididx[0],i]-times[survivors[0,i-1],i-1])
        diff2 = ( freqs[survivors[:,i-2],i-2] - freqs[survivors[:,i-1],i-1] ) / ( times[survivors[0,i-2],i-2] - times[survivors[0,i-1],i-1] )
        diff2 = np.reshape(diff2,(-1,1)) + ( - np.reshape(freqs[survivors[:,i-1],i-1], (-1,1)) + freqs[valididx,i]  ) / ( - times[survivors[0,i-1],i-1] + times[valididx[0],i] )
        dist_metric = np.reshape(path_dist, (-1,1)) - 20*np.log10(mags[valididx,i]+1e-10) + np.minimum(np.abs(diff1), np.abs(np.abs(diff1)-gamma))*c1 #+ diff2**2*c2
        bestidx = np.argpartition(dist_metric.flatten(), n_survivor)[:n_survivor]
        bestidx = np.unravel_index(bestidx, dist_metric.shape)
        new_survivors = np.zeros((n_survivor, i+1))
        for n in range(n_survivor):
            new_survivors[n,:i] = survivors[bestidx[0][n],:i]
            new_survivors[n,-1] = valididx[bestidx[1][n]]
            path_dist[n] = dist_metric[bestidx[0][n], bestidx[1][n]]
        survivors[:,:(i+1)] = new_survivors
        

    k = np.argmin(path_dist)
    return survivors[k]

    







        
