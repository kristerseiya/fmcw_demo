
import numpy as np
import chirpgen
from chirpgen import ChirpGeneratorConfig
from chirpgen import modulation
from filters import SignalModel
from utils import upsample

from scipy.fft import ifft, fft
from scipy.optimize import minimize
from scipy.signal import get_window
from functools import partial
from copy import deepcopy

from typing import Literal, List

def _get_mfs(signal_model:SignalModel, t:np.ndarray, delays:List[float]):
    mfs = []
    for d in delays:
        mfs.append(signal_model.get_complex_waveform(t-d*(t[1]-t[0])))
    mfs = np.stack(mfs)
    return mfs

def _offgrid_objective(y:np.ndarray, signal_model:SignalModel, t:np.ndarray, delays:List[float]):
    mfs = _get_mfs(signal_model, t, delays)
    if len(mfs) == 0:
        quad = np.abs(np.conj(mfs[0]) @ y)**2
    else:
        corr = (np.conj(mfs) @ y)
        quad = np.conj(corr).T @ np.linalg.inv(np.conj(mfs) @ mfs.T) @ corr
        quad = np.abs(quad)
    return -quad

def matching_pursuit(signal_model:SignalModel, 
                    t:np.ndarray, y:np.ndarray, 
                    uprate:int, 
                    refine:bool=True,
                    orthogonalize:bool=False,
                    n_iter:int=100, rel_tol:float=1e-5,
                    get_first_mf_response:bool=False):
    M = uprate
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    delays_coarse = []
    residual = deepcopy(y)
    argmaxes = []
    if not orthogonalize:
        amplitude = []
        delays = []
    print("matching pursuit")    
    for k in range(n_iter):
        residual_up = upsample(residual, M)
        mf_output = np.abs(ifft(fft(residual_up,norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if get_first_mf_response and (k==0):
            mf_output0 = deepcopy(mf_output)
        if len(argmaxes) > 0:
            for i in argmaxes:
                mf_output[i] = 0
        argmax_idx = np.argmax(mf_output)
        argmaxes.append(argmax_idx)
        delays_coarse.append(argmax_idx*1.0/uprate)
        if refine:
            if orthogonalize:
                obj = partial(_offgrid_objective,  y, signal_model, t)
                bounds = [ (a-1/M, a+1/M) for a in delays_coarse]
                res = minimize(obj, delays_coarse, method="Nelder-Mead", bounds=bounds)
                delays = res.x
            else:
                obj = partial(_offgrid_objective,  residual, signal_model, t)
                bounds = [ (delays_coarse[-1]-1/M, delays_coarse[-1]+1/M), ]
                res = minimize(obj, [delays_coarse[-1],], method="Nelder-Mead", bounds=bounds)
                delays.extend(res.x)
        else:
            delays.append(delays_coarse[-1])
        
        if orthogonalize:
            mfs = _get_mfs(signal_model, t, delays)
            if k == 0:
                amplitude = ( np.conj(mfs[0]) @ y ) / np.sum(np.abs(mfs[0])**2)
                residual = y - amplitude * mfs[0]
            else:
                amplitude =  np.linalg.inv(np.conj(mfs) @ mfs.T) @ ( np.conj(mfs) @ y )
                residual = y - amplitude @ mfs
        else:
            mfs = _get_mfs(signal_model, t, [delays[-1],])[0]
            a = ( np.conj(mfs) @ y ) / np.sum(np.abs(mfs)**2)
            amplitude.append(a)
            residual = residual - a * mfs
        residual_power = (np.linalg.norm(residual, ord=2) / np.linalg.norm(y, ord=2))
        print(f"\rIter {k+1:6d} | residual = {residual_power:.3e}", end="", flush=True)
        if residual_power < rel_tol:
            break
    print()
    if get_first_mf_response:
        return np.array(delays), np.array(amplitude), mf_output0
    return np.array(delays), np.array(amplitude)

def _offgrid_objective_bartlett(y:np.ndarray, 
                                signal_model:SignalModel, t:np.ndarray, 
                                window_size:int,
                                delays:List[float]):
    mfs = _get_mfs(signal_model, t, delays)
    if len(mfs) == 0:
        val = 0
        for i in range(0, len(y), window_size):
            val += np.abs(np.conj(mfs[0,i:i+window_size]) @ y[i:i+window_size])**2
    else:
        val = 0
        for i in range(0, len(y), window_size):
            corr = (np.conj(mfs[:,i:i+window_size]) @ y[i:i+window_size])
            quad = np.conj(corr).T @ np.linalg.inv(np.conj(mfs[:,i:i+window_size]) @ mfs[:,i:i+window_size].T) @ corr
            val += np.abs(quad)
    return -val

def bartlett_matching_pursuit(signal_model:SignalModel, 
                                t:np.ndarray, y:np.ndarray, 
                                uprate:int, 
                                window_size: int,
                                refine:bool=True,
                                orthogonalize:bool=False,
                                n_iter:int=100, rel_tol:float=1e-5,
                                get_first_mf_response:bool=False):
    M = uprate
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    delays_coarse = []
    residual = deepcopy(y)
    argmaxes = []
    if not orthogonalize:
        amplitude = []
        delays = []
    print("bartlett matching pursuit")
    for k in range(n_iter):
        residual_up = upsample(residual, M)
        mf_output = np.zeros((len(residual_up),))
        rect = np.arange(len(residual_up))
        for i in range(0, len(residual_up), window_size*M):
            mf_output += np.abs(ifft(fft(residual_up*(rect>=i)*(rect<i+window_size*M),norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        # mf_output = np.abs(ifft(fft(residual_up,norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if get_first_mf_response and (k == 0):
            mf_output0 = deepcopy(mf_output)
        if len(argmaxes) > 0:
            for i in argmaxes:
                mf_output[i] = 0
        argmax_idx = np.argmax(mf_output)
        argmaxes.append(argmax_idx)
        argmax_idx = np.argmax(mf_output)
        delays_coarse.append(argmax_idx*1.0/uprate)
        if refine:
            obj = partial(_offgrid_objective_bartlett,  residual, signal_model, t, window_size)
            if orthogonalize:
                bounds = [ (a-1/M, a+1/M) for a in delays_coarse]
                res = minimize(obj, delays_coarse, method="Nelder-Mead", bounds=bounds)
                delays = res.x
            else:
                bounds = [ (delays_coarse[-1]-1/M, delays_coarse[-1]+1/M), ]
                res = minimize(obj, [delays_coarse[-1],], method="Nelder-Mead", bounds=bounds)
                delays.extend(res.x)
        else:
            delays.append(delays_coarse[-1])

        if orthogonalize:
            mfs = _get_mfs(signal_model, t, delays)
            if k == 0:
                amplitude = np.zeros_like(mfs)
                amplitude[0,i:i+window_size] = ( np.conj(mfs[0,i:i+window_size]) @ y[i:i+window_size] ) / np.sum(np.abs(mfs[0,i:i+window_size])**2)
                residual = y - amplitude * mfs[0]
            else:
                amplitude = np.zeros_like(mfs)
                for i in range(0, len(y), window_size):
                    amplitude[:,i:i+window_size] =  ( np.linalg.inv(np.conj(mfs[:, i:i+window_size]) @ mfs[:, i:i+window_size].T) @ ( np.conj(mfs[:, i:i+window_size]) @ y ) )[:, None]
                residual = y - np.sum(amplitude * mfs,axis=0)
        else:
            mfs = _get_mfs(signal_model, t, [delays[-1],])[0]
            a = np.zeros_like(mfs)
            for i in range(0, len(y), window_size):
                a[i:i+window_size] = ( np.conj(mfs[i:i+window_size]) @ y[i:i+window_size] ) / np.sum(np.abs(mfs[i:i+window_size])**2)
            amplitude.append(a)
            residual = residual - a * mfs
        residual_power = (np.linalg.norm(residual, ord=2) / np.linalg.norm(y, ord=2))
        print(f"\rIter {k+1:6d} | residual = {residual_power:.3e}", end="", flush=True)
        if residual_power < rel_tol:
            break
    print()

    if get_first_mf_response:
        return np.array(delays), np.array(amplitude), mf_output0
    return np.array(delays), np.array(amplitude)

def _compute_brownian_var_cov(N):
    
    pos = np.arange(N)
    var = np.abs(pos)*1.0
    var_minus = np.abs(np.expand_dims(var, axis=1) - np.expand_dims(var, axis=0))
    cov = ( np.abs(np.expand_dims(var, axis=1)) + np.abs(np.expand_dims(var, axis=0)) - var_minus ) / 2

    return var, cov

def _slide_average(arr, window_size):
    new_arr = np.copy(arr)
    for i in range(0, len(arr), window_size):
        new_arr[i:i+window_size] = np.mean(arr[i:i+window_size])
    return new_arr

def _meanify(cov, window_size):
    N = len(cov)
    newcov = np.copy(cov)
    for i in range(0, N, window_size):
        for j in range(0, N, window_size):
            mean_corr2 = np.mean(cov[i:i+window_size,j:j+window_size])
            mean_corr0 = np.mean(cov[i:i+window_size,j:j+window_size],axis=0,keepdims=True)
            mean_corr1 = np.mean(cov[i:i+window_size,j:j+window_size],axis=1,keepdims=True)
            newcov[i:i+window_size,j:j+window_size] = cov[i:i+window_size,j:j+window_size] + mean_corr2 - mean_corr0 - mean_corr1
    return newcov

def compute_bartlett_mcrb(config: ChirpGeneratorConfig, window_size: int, cov):

    sample_rate = config.sample_rate
    Trep = config.Trep
    L = config.phase_noise_var / sample_rate
    sigma2 = config.additive_noise_var**2

    modf = modulation.get_modulation(config)
    t = np.arange(0, Trep, 1/sample_rate)
    f0 = 2*np.pi*modf.generate_freq(t, 0, 0, normalize_freq=False)
    
    N = int(sample_rate*Trep)

    cov = _meanify(cov, window_size)
    var = np.diag(cov)

    var_minus = ( np.expand_dims(var, axis=1) + np.expand_dims(var, axis=0) ) - 2*cov
    var_plus = ( np.expand_dims(var, axis=1) + np.expand_dims(var, axis=0) ) + 2*cov

    decay = np.exp(-1/2*var*L)
    Delta = np.exp(-1/2*var_minus*L) - np.exp(-1/2*var_plus*L)
    
    f = f0 - _slide_average(f0, window_size)
    fnorm = np.linalg.norm(f)
    f = f / fnorm
    mcrb = ( 1/2*(f) @ Delta @ (f) + sigma2) / fnorm**2 / np.mean(decay)**2

    return mcrb/Trep**2

def find_optimal_window_size(config: ChirpGeneratorConfig, max_n_windows: int=30):
    Trep = config.Trep
    sample_rate = config.sample_rate
    N = int(Trep*sample_rate)
    window_size_rate = 1 / np.arange(max_n_windows, 0, -1)
    window_sizes = np.ceil(window_size_rate*N).astype(int)
    window_sizes = np.array(list(set(window_sizes)))
    window_sizes = np.sort(window_sizes)
    mcrbs = np.zeros(len(window_sizes))
    _, cov = _compute_brownian_var_cov(N)
    for i, w in enumerate(window_sizes):
        mcrbs[i] = compute_bartlett_mcrb(config, w, cov)
    return window_sizes[np.argmin(mcrbs)]


def _make_into_subsequences(x, M):
    tmp = np.reshape(np.roll(x, M-1), (-1, M))
    # tmp = np.reshape(x, (-1, M))
    tmp = np.transpose(np.flip(tmp, axis=1))
    return tmp

def _concat_dft_conj(x):
    x_conj = np.zeros_like(x)
    x_conj[...,0] = np.conj(x[...,0])
    x_conj[...,1:] = np.conj(x[...,:0:-1])
    return np.concatenate([x, x_conj], axis=0)

def basis_pursuit(signal_model:SignalModel, 
                  t:np.ndarray, y:np.ndarray, uprate:int,
                  lambd:float, gamma:float=None, 
                  rho:float=10, n_iter:int=100, 
                  regularization:Literal["l1", "elastic-net", "mcp"]="l1", tol=1e-5):

    if (regularization=="elastic-net") and (gamma is None):
        gamma = lambd / 10
    elif (regularization=="mcp") and (gamma is None):
        gamma = 2

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    MF = _make_into_subsequences(mf, M)

    b = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    u = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    b /= np.sqrt(np.sum(np.abs(b)**2))
    u /= np.sqrt(np.sum(np.abs(u)**2))

    MF_DFT = fft(MF, axis=-1)
    # mf_freqres = fft(y*mf[::M],norm="ortho")*np.conj(MF_DFT)
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    # MF_DFT = _concat_dft_conj(MF_DFT)
    # mf_freqres = _concat_dft_conj(mf_freqres)

    # diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # N
    sum_psd = np.sum(np.abs(MF_DFT)**2, axis=0)
    sum_psd[0] = sum_psd[0]*2
    sum_psd[1:] = sum_psd[1:] + sum_psd[:0:-1]
    diags = 1 + 1/rho*sum_psd # N
    
    a_old = np.zeros_like(b)
    print("basis pursuit")
    for i_iter in range(n_iter):
        tmp = mf_freqres + rho * fft(b-u, axis=-1, norm="ortho") # M x N
        tmp2 = np.sum(MF_DFT * tmp, axis=0)
        tmp2[0] = tmp2[0]*2
        tmp2[1:] = tmp2[1:] + np.conj(tmp2[:0:-1])
        tmp = 1/rho*tmp - 1/rho**2 * tmp2 / diags * np.conj(MF_DFT) # M x N
        a = ifft(tmp, axis=-1, norm="ortho")
        if regularization == "l1":
            b = np.exp(1j*np.angle(a+u)) * np.maximum(np.abs(a+u)-lambd/rho, np.zeros_like(a))
        elif regularization == "elastic-net":
            b = np.exp(1j*np.angle(a+u)) * np.maximum(np.abs(a+u)-lambd/rho, np.zeros_like(a)) * (rho**2 / (rho**2 + lambd*gamma))
        elif regularization == "mcp":
            b = np.zeros_like(b)
            aumag = np.abs(a+u)
            r1 = (aumag>(lambd/rho))*(aumag<=(gamma*lambd/rho))
            r2 = (aumag>gamma*lambd/rho)
            b[r1] = np.exp(1j*np.angle(a[r1]+u[r1])) * (aumag[r1]-lambd/rho) / (1 - 1/gamma)
            b[r2] = a[r2]+u[r2]
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    print()
    a = np.transpose(a, (1, 0))
    a = a.flatten()
    return a

def _strided_windows_v2(X, L, s, window=None, fftbins=True):
    X = np.atleast_2d(np.asarray(X))
    K, N = X.shape
    if L > N:
        raise ValueError("L must be <= X.shape[1]")
    n_win = (N - L) // s + 1
    starts = np.arange(n_win) * s

    if window is None:
        seg = lambda n: 1
    elif callable(window):
        seg = window                         # e.g. lambda n: windows.kaiser(n, 8)
    else:
        seg = lambda n: get_window(window, n, fftbins=fftbins)

    W = np.zeros((n_win, N), dtype=X.dtype if window is None else float)
    w_L = seg(L)
    for w in range(n_win - 1):
        W[w, starts[w]:starts[w] + L] = w_L
    W[-1, starts[-1]:] = seg(N - starts[-1])   # last segment may be longer

    return W[:, None, :] * X[None, :, :]       # (n_win, K, N)

def _make_segmented_subsequences(x, M, segment_len, stride, window=None):
    tmp = np.reshape(np.roll(x, M-1), (-1, M))
    tmp = np.transpose(np.flip(tmp, axis=1))
    return _strided_windows_v2(tmp, segment_len, stride, window)

def bartlett_basis_pursuit(signal_model:SignalModel, 
                           t:np.ndarray, y:np.ndarray, uprate:int,
                           lambd:float, window_len:int, stride:int, 
                           rho:float=10, window=None, n_iter=100, tol=1e-5):

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    MF = _make_into_subsequences(mf, M)

    MF = _make_segmented_subsequences(mf, M, window_len, stride, window)
    MF_DFT = fft(MF, axis=-1)

    L = MF.shape[0]

    MF_DFT = MF_DFT.reshape((L*M, N)) # L*M x N
    
    b = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    u = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    b /= np.sqrt(np.sum(np.abs(b)**2, axis=(1,2), keepdims=True))
    u /= np.sqrt(np.sum(np.abs(u)**2, axis=(1,2), keepdims=True))

    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # 
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    print("bartlett basis pursuit")
    for i_iter in range(n_iter):
        tmp = mf_freqres + rho * fft(np.reshape(b-u, (L*M, N)), axis=-1, norm="ortho") # L*M x N
        tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # L*M x N
        tmp = ifft(tmp, axis=-1, norm="ortho") # L*M x N
        a = np.reshape(tmp, (L, M, N)) # L x M x N
        aunorm = np.linalg.norm(a+u, axis=0, ord=2, keepdims=True)
        b = (a+u) * np.maximum(1-lambd/rho/aunorm, np.zeros_like(a))
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    print()
    a = np.transpose(a, (0, 2, 1))
    a = np.reshape(a, (L, N*M))
    return a

def _aggregate(p, shape, group_len, group_stride):
    p_aggregated = np.zeros(shape, dtype=p.dtype)
    cnt = np.zeros((shape[1],))
    for i in range(0, p.shape[1]):
        p_aggregated[:, (i*group_stride):(i*group_stride+group_len)] += p[:,i][:,None]
        cnt[(i*group_stride):(i*group_stride+group_len)] += 1
    p_aggregated /= cnt
    return p_aggregated

def _align_directions(p):
    for i in range(1,p.shape[1]):
        p[:,i] = np.exp(1j*np.angle(np.conj(p[:,i]) @ p[:,i-1])) * p[:,i]
    return p

def _normalize_columns(p):
    for i in range(p.shape[1]):
        pnorm = np.sqrt(np.sum(np.abs(p[:,i])**2))
        if pnorm > 1e-3:
            p[:,i] = p[:,i] / pnorm
    return p

def bartlett_low_rank_basis_pursuit(signal_model:SignalModel, 
                                    t:np.ndarray, y:np.ndarray, uprate:int,
                                    lambd:float, window_len:int, stride:int,
                                    group_len:int, group_stride:int,
                                    rho:float=10, window=None, n_iter=100, tol=1e-5):

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)

    MF = _make_segmented_subsequences(mf, M, window_len, stride, window)
    MF_DFT = fft(MF, axis=-1)

    K = (M*N-group_len) // group_stride + 1
    K = len(np.arange(0, N*M-group_len+group_stride, group_stride))

    L = MF.shape[0]

    MF_DFT = MF_DFT.reshape((L*M, N)) # L*M x N
    
    x = np.random.rand(N*M) * np.exp(1j * 2*np.pi* np.random.rand(M*N))
    p = np.random.rand(L, K) * np.exp(1j *2*np.pi* np.random.rand(L, K))
    p = _normalize_columns(p)
    p = _align_directions(p)
    # p_init = p.copy()
    #p = _normalize_columns(p)
    p_aggregated = _aggregate(p, (L,N*M), group_len, group_stride)
    #p_aggregated = _normalize_columns(p_aggregated)
    b = p_aggregated * x
    # b = np.random.rand(L, N*M) * np.exp(1j * 2*np.pi* np.random.rand(L, N*M))
    # b /= np.sqrt(np.sum(np.abs(b)**2, axis=0, keepdims=True))
    # p_init = p.copy()
    # p_aggregated_init = p_aggregated.copy()
    # x_init = x.copy()
    # b_init = b.copy()

    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # 
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)

    a_old = np.zeros_like(b)
    print("bartlett low rank basis pursuit")
    for i_iter in range(n_iter):
        b = np.reshape(b, (L, N, M))
        b = np.transpose(b, (0, 2, 1))
        b = np.reshape(b, (L*M, N))

        tmp = mf_freqres + rho * fft(b, axis=-1, norm="ortho") # L*M x N
        tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # L*M x N
        a = ifft(tmp, axis=-1, norm="ortho") # L*M x N

        a = np.reshape(a, (L, M, N))
        a = np.transpose(a, (0, 2, 1))
        a = np.reshape(a, (L, N*M))

        x = np.sum( a * np.conj(p_aggregated), axis=0 )
        x = np.exp(1j*np.angle(x))*np.maximum(np.abs(x)-lambd/rho, np.zeros_like(x))
        for k, i in enumerate(range(0, N*M-group_len+group_stride, group_stride)):
            p[:,k] = a[:, i:i+group_len] @ np.conj(x[i:i+group_len])
        p = _normalize_columns(p)
        p = _align_directions(p)
        #p = _normalize_columns(p)
        p_aggregated = _aggregate(p, (L,N*M), group_len, group_stride)
        #p_aggregated = _normalize_columns(p_aggregated)
        b = p_aggregated * x
        diff = np.mean(np.abs(a-a_old))
        print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    print()
    # a = np.reshape(a, (L, M, N))
    # a = np.transpose(a, (0, 2, 1))
    # a = np.reshape(a, (L, N*M))
    return a#, b, p, p_aggregated, x, p_init, p_aggregated_init, x_init


def bartlett_basis_pursuit_multiresolution(mf:np.ndarray, y:np.ndarray, 
                                           lambd:float, window_len:List[int], stride:List[int], 
                                           rho:float=10, window=None, iter=100, tol=1e-5):

    N = len(y)
    M = int(len(mf)/len(y))

    MF = _make_segmented_subsequences(mf, M, window_len, stride, window)
    MF_DFT = fft(MF, axis=-1)

    L = MF.shape[0]

    MF_DFT = MF_DFT.reshape((L*M, N)) # L*M x N
    
    b = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    u = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    b /= np.sqrt(np.sum(np.abs(b)**2, axis=(1,2), keepdims=True))
    u /= np.sqrt(np.sum(np.abs(u)**2, axis=(1,2), keepdims=True))

    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # 
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    print("bartlett basis pursuit")
    for i_iter in range(iter):
        tmp = mf_freqres + rho * fft(np.reshape(b-u, (L*M, N)), axis=-1, norm="ortho") # L*M x N
        tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # L*M x N
        tmp = ifft(tmp, axis=-1, norm="ortho") # L*M x N
        a = np.reshape(tmp, (L, M, N)) # L x M x N
        aunorm = np.linalg.norm(a+u, axis=0, ord=2, keepdims=True)
        b = (a+u) * np.maximum(1-lambd/rho/aunorm, np.zeros_like(a))
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    print()
    a = np.transpose(a, (0, 2, 1))
    a = np.reshape(a, (L, N*M))
    return a