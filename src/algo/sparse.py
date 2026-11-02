
import numpy as np
import chirpgen
from chirpgen import ChirpGeneratorConfig
from chirpgen import modulation
from filters import SignalModel
from utils import upsample

from scipy.fft import ifft, fft
from scipy.optimize import minimize
from scipy.signal import get_window
import scipy.linalg as spla
from functools import partial
from copy import deepcopy

from typing import Literal, List

VERBOSE = False

def _get_mfs(signal_model:SignalModel, t:np.ndarray, delays:List[float]):
    mfs = []
    for d in delays:
        mfs.append(signal_model.get_complex_waveform(t-d*(t[1]-t[0])))
    mfs = np.stack(mfs)
    return mfs

def _offgrid_objective(y:np.ndarray, signal_model:SignalModel, t:np.ndarray, delays:List[float]):
    mfs = _get_mfs(signal_model, t, delays)
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
                    get_first_mf_response:bool=False, 
                    t_low:float=None, t_high:float=None,
                    verbose:bool=False):
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

    apply_mask = True if t_low or t_high else False
    if apply_mask:
        mask = np.ones((len(mf),), dtype=bool)
        if t_low:
            mask *= (t_fine>=t_low)
        if t_high:
            mask *= (t_fine<=t_high)
        idx0 = np.argmax(mask)

    if verbose or VERBOSE:
        print("matching pursuit")    
    for k in range(n_iter):
        residual_up = upsample(residual, M)
        mf_output = np.abs(ifft(fft(residual_up,norm="ortho")*np.conj(fft(mf)),norm="ortho"))**2 / len(y)
        if get_first_mf_response and (k==0):
            mf_output0 = deepcopy(mf_output)
        if len(argmaxes) > 0:
            for i in argmaxes:
                mf_output[i] = 0
        if apply_mask:
            argmax_idx = np.argmax(mf_output)
        else:
            argmax_idx = np.argmax(mf_output[mask]) + idx0
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
            amplitude =  np.linalg.inv(np.conj(mfs) @ mfs.T) @ ( np.conj(mfs) @ y )
            residual = y - amplitude @ mfs
        else:
            mfs = _get_mfs(signal_model, t, [delays[-1],])[0]
            a = ( np.conj(mfs) @ y ) / np.sum(np.abs(mfs)**2)
            amplitude.append(a)
            residual = residual - a * mfs
        residual_power = (np.linalg.norm(residual, ord=2) / np.linalg.norm(y, ord=2))
        if verbose or VERBOSE:
            print(f"\rIter {k+1:6d} | residual = {residual_power:.3e}", end="", flush=True)
        if residual_power < rel_tol:
            break
    if verbose or VERBOSE:
        print()
    if get_first_mf_response:
        return np.array(delays), np.array(amplitude), mf_output0
    return np.array(delays), np.array(amplitude)

def _get_start_end_of_each_window(N, L, s):
    n_win = (N - L) // s + 1
    starts = np.arange(n_win) * s
    ends = np.concatenate([ starts[:-1]+L, [N,] ])
    return starts, ends

def _offgrid_objective_bartlett(y:np.ndarray, 
                                signal_model:SignalModel, t:np.ndarray, 
                                window_size:int,
                                delays:List[float]):
    mfs = _get_mfs(signal_model, t, delays)
    sidx, eidx = _get_start_end_of_each_window(len(y), window_size, window_size)
    val = 0
    for si, ei in zip(sidx, eidx):
        corr = (np.conj(mfs[:,si:ei]) @ y[si:ei])
        quad = np.conj(corr).T @ np.linalg.inv(np.conj(mfs[:,si:ei]) @ mfs[:,si:ei].T) @ corr
        val += np.abs(quad)
    return -val

def bartlett_matching_pursuit(signal_model:SignalModel, 
                                t:np.ndarray, y:np.ndarray, 
                                uprate:int, 
                                window_len: int,
                                refine:bool=True,
                                orthogonalize:bool=False,
                                n_iter:int=100, rel_tol:float=1e-5,
                                get_first_mf_response:bool=False,
                                t_low:float=None, t_high:float=None,
                                verbose:bool=False):
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

    apply_mask = True if t_low or t_high else False
    if apply_mask:
        mask = np.ones((len(mf),), dtype=bool)
        if t_low:
            mask *= (t_fine>=t_low)
        if t_high:
            mask *= (t_fine<=t_high)
        idx0 = np.argmax(mask)

    sidx, eidx = _get_start_end_of_each_window(len(y), window_len, window_len)
    if verbose or VERBOSE:
        print("bartlett matching pursuit")
    for k in range(n_iter):
        residual_up = upsample(residual, M)
        mf_output = np.zeros((len(residual_up),))
        # rect = np.arange(len(residual_up))
        # for i in range(0, len(residual_up), window_len*M):
        #     mf_output += np.abs(ifft(fft(residual_up*(rect>=i)*(rect<i+window_len*M),norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        for si, ei in zip(sidx, eidx):
            rect = np.zeros_like(residual_up)
            rect[si*M:ei*M] = 1
            mf_output += np.abs(ifft(fft(residual_up*rect,norm="ortho")*np.conj(fft(mf)),norm="ortho"))**2 * np.ceil((ei-si)/M)
        mf_output /= len(y)
        
        if get_first_mf_response and (k == 0):
            mf_output0 = deepcopy(mf_output)
        if len(argmaxes) > 0:
            for i in argmaxes:
                mf_output[i] = 0
        if apply_mask:
            argmax_idx = np.argmax(mf_output)
        else:
            argmax_idx = np.argmax(mf_output[mask]) + idx0
        argmaxes.append(argmax_idx)
        argmax_idx = np.argmax(mf_output)
        delays_coarse.append(argmax_idx*1.0/uprate)
        if refine:
            obj = partial(_offgrid_objective_bartlett,  residual, signal_model, t, window_len)
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
            amplitude = np.zeros_like(mfs)
            for si, ei in zip(sidx, eidx):
                amplitude[:,si:ei] =  ( np.linalg.inv(np.conj(mfs[:, si:ei]) @ mfs[:, si:ei].T) @ ( np.conj(mfs[:, si:ei]) @ y[si:ei] ) )[:, None]
            residual = y - np.sum(amplitude * mfs,axis=0)
        else:
            mfs = _get_mfs(signal_model, t, [delays[-1],])[0]
            a = np.zeros_like(mfs)
            for si, ei in zip(sidx, eidx):
                a[si:ei] = ( np.conj(mfs[si:ei]) @ y[si:ei] ) / np.sum(np.abs(mfs[si:ei])**2)
            amplitude.append(a)
            residual = residual - a * mfs
        residual_power = (np.linalg.norm(residual, ord=2) / np.linalg.norm(y, ord=2))
        if verbose or VERBOSE:
            print(f"\rIter {k+1:6d} | residual = {residual_power:.3e}", end="", flush=True)
        if residual_power < rel_tol:
            break
    if verbose or VERBOSE:
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


def _contrsuct_submatrix_of_circulant(x, minshift, maxshift, downsample=1):
    
    assert(maxshift>=minshift)

    if (minshift == 0) and (maxshift == len(x)-1):
        X = spla.circulant(x)[::downsample,:]
    else:
        N = (len(x) + downsample - 1) // downsample 
        X = np.zeros((N, maxshift-minshift+1))
        x_rolled = np.roll(x, (minshift,))
        X[:,0] = x_rolled[::downsample]
        for i in range(maxshift-minshift):
            x_rolled = np.roll(x_rolled, (1,))
            X[:,i+1] = x_rolled[::downsample]
    return X

def basis_pursuit(signal_model:SignalModel, 
                  t:np.ndarray, y:np.ndarray, uprate:int,
                  lambd:float, gamma:float=None, 
                  rho:float=10, real_valued:bool=False,
                  n_iter:int=100, 
                  regularization:Literal["l1", "elastic-net", "mcp"]="l1", 
                  tol:float=1e-5,
                  t_low:float=None, t_high:float=None,
                  verbose:bool=False):
    
    if (regularization=="elastic-net") and (gamma is None):
        gamma = lambd / 10
    elif (regularization=="mcp") and (gamma is None):
        gamma = 2

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)

    minshift = 0 if t_low is None else int(np.ceil(t_low / del_t * M))
    maxshift = len(mf)-1 if t_high is None else int(np.floor(t_high / del_t * M))

    X = _contrsuct_submatrix_of_circulant(mf, minshift, maxshift, M)
    if verbose or VERBOSE:
        print("basis pursuit")
        print(f"There are {X.shape[1]:d} number of columns in X")

    b = np.random.rand(M*len(t)) * np.exp(1j * 2*np.pi* np.random.rand(M*len(t)))
    u = np.random.rand(M*len(t)) * np.exp(1j * 2*np.pi* np.random.rand(M*len(t)))
    b /= np.sqrt(np.sum(np.abs(b)**2))
    u /= np.sqrt(np.sum(np.abs(u)**2))
    XhX = np.linalg.inv(np.conj(X).T @ X + rho * np.eye(M*len(t)))
    a_old = np.zeros_like(b)
    for i_iter in range(n_iter):
        a = XhX @ (np.conj(X).T @ y + rho * ( b - u ))
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
        if verbose or VERBOSE:
            print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    return a

def _strided_windows(X, L, s, window=None, fftbins=True):
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

def _contrsuct_submatrix_of_circulant_2(x, window_len, stride, window, 
                                        minshift, maxshift, downsample=1):
    
    assert(maxshift>=minshift)
    xsub = _strided_windows(x, window_len, stride, window)[:,0,:]
    L = len(xsub)
    N = (len(x) + downsample - 1) // downsample 

    if (minshift == 0) and (maxshift == len(x)-1):
        X = spla.circulant(xsub)[:,::downsample,:]
        X = np.transpose(X, (1,2,0))
        X = np.reshape(X, (N, len(x)*L))
    else:
        xsub = xsub.transpose()
        X = np.zeros((N, (maxshift-minshift+1)*L))
        x_rolled = np.roll(xsub, (minshift,0))
        X[:,:L] = x_rolled[::downsample]
        for i in range(maxshift-minshift):
            x_rolled = np.roll(x_rolled, (1,0))
            X[:,(i+1)*L:(i+2)*L] = x_rolled[::downsample]

    return X

def bartlett_basis_pursuit(signal_model:SignalModel, 
                           t:np.ndarray, y:np.ndarray, uprate:int,
                           lambd:float, 
                           window_len:int, stride:int=None,
                           rho:float=10, window=None, 
                           n_iter:int=100, 
                           tol:float=1e-5,
                           t_low:float=None, t_high:float=None,
                           verbose:bool=False):

    if stride is None:
        stride = window_len // 2
    assert(window_len >= stride)

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)

    minshift = 0 if t_low is None else int(np.ceil(t_low / del_t * M))
    maxshift = len(mf)-1 if t_high is None else int(np.floor(t_high / del_t * M))

    X = _contrsuct_submatrix_of_circulant_2(mf, window_len, stride, window, minshift, maxshift, M)
    L = X.shape[1] // len(mf)
    if verbose or VERBOSE:
        print("bartlett basis pursuit")
        print(f"There are {X.shape[1]:d} number of columns in X")
    
    b = np.random.rand(N*M, L) * np.exp(1j * 2*np.pi* np.random.rand(N*M, L))
    u = np.random.rand(N*M, L) * np.exp(1j * 2*np.pi* np.random.rand(N*M, L))
    b /= np.sqrt(np.sum(np.abs(b)**2, axis=0, keepdims=True))
    u /= np.sqrt(np.sum(np.abs(u)**2, axis=0, keepdims=True))

    XhX = np.linalg.inv(np.conj(X).T @ X + rho * np.eye(M*L*len(t)))
    a_old = np.zeros_like(b)
    for i_iter in range(n_iter):
        a = XhX @ (np.conj(X).T @ y + rho * ( b - u ))
        a = np.reshape(a, (N*M, L))
        aunorm = np.linalg.norm(a+u, axis=1, ord=2, keepdims=True)
        b = (a+u) * np.maximum(1-lambd/rho/aunorm, np.zeros_like(a))
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        if verbose or VERBOSE:
            print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    if verbose or VERBOSE:
        print()
    a = np.transpose(a, (0, 1))
    return a

def _make_into_subsequences(x, M):
    tmp = np.reshape(np.roll(x, M-1), (-1, M))
    # tmp = np.reshape(x, (-1, M))
    tmp = np.transpose(np.flip(tmp, axis=1))
    return tmp

def basis_pursuit_faster(signal_model:SignalModel, 
                  t:np.ndarray, y:np.ndarray, uprate:int,
                  lambd:float, gamma:float=None, 
                  rho:float=10, real_valued:bool=False,
                  n_iter:int=100, 
                  regularization:Literal["l1", "elastic-net", "mcp"]="l1", 
                  tol:float=1e-5,
                  t_low:float=None, t_high:float=None,
                  verbose:bool=False):

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

    apply_mask = True if t_low or t_high else False
    if apply_mask:
        mask = np.ones_like(mf)
        if t_low:
            mask *= (t_fine>=t_low)
        if t_high:
            mask *= (t_fine<=t_high)
        mask = np.reshape(mask, (N, M))
        mask = mask.transpose()

    b = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    u = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    if apply_mask:
        b *= mask
    b /= np.sqrt(np.sum(np.abs(b)**2))
    u /= np.sqrt(np.sum(np.abs(u)**2))

    MF_DFT = fft(MF, axis=-1)
    sum_psd = np.sum(np.abs(MF_DFT)**2, axis=0)
    if real_valued:
        sum_psd[0] = sum_psd[0]*2
        sum_psd[1:] = sum_psd[1:] + sum_psd[:0:-1]
    diags = 1 + 1/rho*sum_psd
    # diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # N
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    if verbose or VERBOSE:
        print("basis pursuit (faster)")
    for i_iter in range(n_iter):
        tmp = mf_freqres + rho * fft(b-u, axis=-1, norm="ortho") # M x N
        tmp2 = np.sum(MF_DFT * tmp, axis=0)
        if real_valued:
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
        if apply_mask:
            b *= mask
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        if verbose or VERBOSE:
            print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    if verbose or VERBOSE:
        print()
    a = np.transpose(a, (1, 0))
    a = a.flatten()
    return a

def _make_segmented_subsequences(x, M, segment_len, stride, window=None):
    tmp = np.reshape(np.roll(x, M-1), (-1, M))
    tmp = np.transpose(np.flip(tmp, axis=1))
    return _strided_windows(tmp, segment_len, stride, window)

def bartlett_basis_pursuit_faster(signal_model:SignalModel, 
                           t:np.ndarray, y:np.ndarray, uprate:int,
                           lambd:float, 
                           window_len:int, stride:int=None,
                           rho:float=10, window=None, 
                           n_iter:int=100, 
                           tol:float=1e-5,
                           t_low:float=None, t_high:float=None,
                           verbose:bool=False):

    if stride is None:
        stride = window_len // 2
    assert(window_len >= stride)

    M = uprate
    N = len(y)
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    MF = _make_segmented_subsequences(mf, M, window_len, stride, window)
    MF_DFT = fft(MF, axis=-1)

    L = MF.shape[0]

    apply_mask = True if t_low or t_high else False
    if apply_mask:
        mask = np.ones_like(mf)
        if t_low:
            mask *= (t_fine>=t_low)
        if t_high:
            mask *= (t_fine<=t_high)
        mask = np.reshape(mask, (N, M))
        mask = mask.transpose()

    MF_DFT = MF_DFT.reshape((L*M, N)) # L*M x N
    
    b = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    u = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
    if apply_mask:
        b *= mask
    b /= np.sqrt(np.sum(np.abs(b)**2, axis=(1,2), keepdims=True))
    u /= np.sqrt(np.sum(np.abs(u)**2, axis=(1,2), keepdims=True))

    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # 
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    if verbose or VERBOSE:
        print("bartlett basis pursuit (faster)")
    for i_iter in range(n_iter):
        tmp = mf_freqres + rho * fft(np.reshape(b-u, (L*M, N)), axis=-1, norm="ortho") # L*M x N
        tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # L*M x N
        tmp = ifft(tmp, axis=-1, norm="ortho") # L*M x N
        a = np.reshape(tmp, (L, M, N)) # L x M x N
        aunorm = np.linalg.norm(a+u, axis=0, ord=2, keepdims=True)
        b = (a+u) * np.maximum(1-lambd/rho/aunorm, np.zeros_like(a))
        if apply_mask:
            b *= mask
        u = u + a - b
        diff = np.mean(np.abs(a-a_old))
        if verbose or VERBOSE:
            print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    if verbose or VERBOSE:
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

def bartlett_low_rank_basis_pursuit_faster(signal_model:SignalModel, 
                                    t:np.ndarray, y:np.ndarray, uprate:int,
                                    lambd:float, 
                                    window_len:int, group_len:int, 
                                    stride:int=None, group_stride:int=None,
                                    rho:float=10, window=None, 
                                    n_iter:int=100, tol:float=1e-5,
                                    t_low:float=None, t_high:float=None,
                                    verbose:bool=False):

    if stride is None:
        stride = window_len // 2
    assert(window_len >= stride)
    if group_stride is None:
        group_stride = group_len // 2
    assert(group_len >= group_stride)

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

    apply_mask = True if t_low or t_high else False
    if apply_mask:
        mask = np.ones_like(mf)
        if t_low:
            mask *= (t_fine>=t_low)
        if t_high:
            mask *= (t_fine<=t_high)
    
    x = np.random.rand(N*M) * np.exp(1j * 2*np.pi* np.random.rand(N*M))
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
    if verbose or VERBOSE:
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
        if apply_mask:
            x *= mask
        for k, i in enumerate(range(0, N*M-group_len+group_stride, group_stride)):
            p[:,k] = a[:, i:i+group_len] @ np.conj(x[i:i+group_len])
        p = _normalize_columns(p)
        p = _align_directions(p)
        #p = _normalize_columns(p)
        p_aggregated = _aggregate(p, (L,N*M), group_len, group_stride)
        #p_aggregated = _normalize_columns(p_aggregated)
        b = p_aggregated * x
        diff = np.mean(np.abs(a-a_old))
        if verbose or VERBOSE:
            print(f"\rIter {i_iter+1:6d} | diff = {diff:.3e}", end="", flush=True)
        if diff < tol:
            break
        a_old = a.copy()
    if verbose or VERBOSE:
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