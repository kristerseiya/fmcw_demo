
import numpy as np
import chirpgen
from chirpgen import ChirpGeneratorConfig
from chirpgen import modulation
import filters
from utils import upsample

from scipy.fft import ifft, fft
from scipy.optimize import minimize
from scipy.signal import get_window
from scipy.sparse.linalg import svds

from typing import Literal, List

def offgrid_orthogonal_matching_pursuit(config, t, y, K):
    modf = chirpgen.modulation.get_modulation(config)
    sample_rate = config.sample_rate
    bandwidth = config.bandwidth
    T = config.Trep
    M = np.ceil(bandwidth/sample_rate).astype(int)*10
    t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
    mf = filters.construct_matchfilter(t_fine, config, normalize=False)
    normalized_delays_coarse = np.zeros((K,))
    normalized_delays = np.zeros((K,))
    amplitude = np.zeros((K,), dtype=np.complex128)
    residual = np.copy(y)
    argmax_idx = np.array([], dtype=int)
    for k in range(K):
        residual_up = upsample(residual, M)
        mf_output = np.abs(ifft(fft(residual_up,norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if len(argmax_idx) > 0:
            mf_output[argmax_idx] = 0
        argmax_idx = np.concatenate([argmax_idx, [np.argmax(mf_output)]])
        normalized_delays_coarse[k] = t_fine[argmax_idx[-1]]/T
        normalized_delays[k] = t_fine[argmax_idx[-1]]/T
        if k == 0:
            def obj(normalized_tau):
                tmp = modf.generate_waveform(t, normalized_tau*T*3e8/2)
                return -np.abs(np.conj(tmp) @ y)
        else:
            def obj(normalized_tau):
                tmp = modf.generate_waveform(t, normalized_tau*T*3e8/2)
                tmp2 = np.conj(tmp) @ y
                tmp3 = np.conj(tmp2).T @ np.linalg.inv(np.conj(tmp) @ tmp.T) @ tmp2
                return -np.abs(tmp3)
        bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays_coarse[:(k+1)]]
        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays[:(k+1)]]
        res = minimize(obj, normalized_delays[:(k+1)], method="Nelder-Mead", bounds=bounds)
        normalized_delays[:(k+1)] = res.x
        
        Xk = modf.generate_waveform(t, normalized_delays[:(k+1)]*T*3e8/2)
        if k == 0:
            amplitude[0] = ( np.conj(Xk) @ y ) / np.sum(np.abs(Xk))
            residual = y - amplitude[0] * Xk
        else:
            amplitude[:(k+1)] = np.linalg.inv(np.conj(Xk) @ Xk.T) @ ( np.conj(Xk) @ y )
            residual = y - amplitude[:(k+1)] @ Xk
    return normalized_delays, amplitude


def offgrid_orthogonal_matching_pursuit_bartlett(config, t, y, K, window_size, get_intermediates=False):
    modf = chirpgen.modulation.get_modulation(config)
    sample_rate = config.sample_rate
    bandwidth = config.bandwidth
    T = config.Trep
    M = np.ceil(bandwidth/sample_rate).astype(int)*10
    t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
    mf = filters.construct_matchfilter(t_fine, config, normalize=False)
    normalized_delays_coarse = np.zeros((K,))
    normalized_delays = np.zeros((K,))
    amplitude = np.zeros((K,len(y)), dtype=np.complex128)
    residual = np.copy(y)
    if get_intermediates:
        intermediates = []
    for k in range(K):
        residual_up = upsample(residual, M)
        mf_output = np.zeros((len(residual_up),))
        rect = np.arange(len(residual_up))
        for i in range(0, len(residual_up), window_size*M):
            mf_output += np.abs(ifft(fft(residual_up*(rect>=i)*(rect<i+window_size*M),norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if get_intermediates:
            intermediates.append(mf_output)
        normalized_delays_coarse[k] = t_fine[np.argmax(mf_output)]/T
        normalized_delays[k] = t_fine[np.argmax(mf_output)]/T
        if k == 0:
            def obj(normalized_tau):
                tmp = modf.generate_waveform(t, normalized_tau*T*3e8/2)
                val = 0
                for i in range(0, len(y), window_size):
                    val += -np.abs(np.conj(tmp[i:i+window_size]) @ y[i:i+window_size])
                return val
        else:
            def obj(normalized_tau):
                tmp = modf.generate_waveform(t, normalized_tau*T*3e8/2)
                val = 0
                for i in range(0, len(y), window_size):
                    tmp2 = np.conj(tmp[:, i:i+window_size]) @ y[i:i+window_size]
                    tmp3 = np.conj(tmp2).T @ np.linalg.inv(np.conj(tmp[:, i:i+window_size]) @ tmp[:, i:i+window_size].T) @ tmp2
                    val += -np.abs(tmp3)
                return val
        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays_coarse[:(k+1)]]
        bounds = [ (a-1/window_size/M, a+1/window_size/M) for a in normalized_delays_coarse[:(k+1)]]
        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays[:(k+1)]]
        res = minimize(obj, normalized_delays[:(k+1)], method="Nelder-Mead", bounds=bounds)
        normalized_delays[:(k+1)] = res.x
        
        Xk = modf.generate_waveform(t, normalized_delays[:(k+1)]*T*3e8/2)
        if k == 0:
            for i in range(0, len(y), window_size):
                amplitude[0,i:i+window_size] = ( np.conj(Xk[i:i+window_size]) @ y[i:i+window_size] ) / np.sum(np.abs(Xk[i:i+window_size]))
            
            residual = y - amplitude[0] * Xk
        else:
            for i in range(0, len(y), window_size):
                amplitude[:(k+1),i:i+window_size] = np.expand_dims( np.linalg.inv(np.conj(Xk[:, i:i+window_size]) @ Xk[:, i:i+window_size].T) @ ( np.conj(Xk[:, i:i+window_size]) @ y[i:i+window_size] ), axis=1 )
            residual = y - np.sum(amplitude[:(k+1)] * Xk, axis=0)
    if get_intermediates:
        return normalized_delays, amplitude, intermediates
    return normalized_delays, amplitude


def offgrid_matching_pursuit(config, t, y, K, get_intermediates=False):
    modf = chirpgen.modulation.get_modulation(config)
    sample_rate = config.sample_rate
    bandwidth = config.bandwidth
    T = config.Trep
    M = np.ceil(bandwidth/sample_rate).astype(int)*10
    t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
    mf = filters.construct_matchfilter(t_fine, config, normalize=False)
    normalized_delays_coarse = np.zeros((K,))
    normalized_delays = np.zeros((K,))
    amplitude = np.zeros((K,), dtype=np.complex128)
    residual = np.copy(y)
    if get_intermediates:
        intermediates = []
    for k in range(K):
        residual_up = upsample(residual, M)
        mf_output = np.abs(ifft(fft(residual_up,norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if get_intermediates:
            intermediates.append(mf_output)
        normalized_delays_coarse[k] = t_fine[np.argmax(mf_output)]/T
        normalized_delays[k] = t_fine[np.argmax(mf_output)]/T
        def obj(normalized_tau):
            tmp = modf.generate_waveform(t, normalized_tau[0]*T*3e8/2)
            return -np.abs(np.conj(tmp) @ y)
        bounds = [(normalized_delays_coarse[k]-1/len(t)/M, normalized_delays_coarse[k]+1/len(t)/M) ]
        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays[:(k+1)]]
        res = minimize(obj, [normalized_delays[k]], method="Nelder-Mead", bounds=bounds)
        normalized_delays[k] = res.x[0]
        
        Xk = modf.generate_waveform(t, normalized_delays[k]*T*3e8/2)
        amplitude[k] = ( np.conj(Xk) @ residual ) / np.sum(np.abs(Xk))
        residual = residual - amplitude[k] * Xk
    if get_intermediates:
        return normalized_delays, amplitude, intermediates
    return normalized_delays, amplitude


def offgrid_matching_pursuit_bartlett(config, t, y, K, window_size, get_intermediates=False):
    modf = chirpgen.modulation.get_modulation(config)
    sample_rate = config.sample_rate
    bandwidth = config.bandwidth
    T = config.Trep
    M = np.ceil(bandwidth/sample_rate).astype(int)*10
    t_fine = np.linspace(t[0],t[-1]+(M-1)/sample_rate/M, M*len(t))
    mf = filters.construct_matchfilter(t_fine, config, normalize=False)
    normalized_delays_coarse = np.zeros((K,))
    normalized_delays = np.zeros((K,))
    amplitude = np.zeros((K,len(y)), dtype=np.complex128)
    residual = np.copy(y)
    if get_intermediates:
        intermediates = []
    for k in range(K):
        residual_up = upsample(residual, M)
        mf_output = np.zeros((len(residual_up),))
        rect = np.arange(len(residual_up))
        for i in range(0, len(residual_up), window_size*M):
            mf_output += np.abs(ifft(fft(residual_up*(rect>=i)*(rect<i+window_size*M),norm="ortho")*np.conj(fft(mf)),norm="ortho")) / len(y)
        if get_intermediates and (k == 0):
            intermediates.append(mf_output)
        normalized_delays_coarse[k] = t_fine[np.argmax(mf_output)]/T
        normalized_delays[k] = t_fine[np.argmax(mf_output)]/T

        def obj(normalized_tau):
            tmp = modf.generate_waveform(t, normalized_tau[0]*T*3e8/2)
            val = 0
            for i in range(0, len(y), window_size):
                val += -np.abs(np.conj(tmp[i:i+window_size]) @ y[i:i+window_size])
            return val

        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays_coarse[:(k+1)]]
        bounds = [ (normalized_delays_coarse[k]-1/window_size/M, normalized_delays_coarse[k]+1/window_size/M) ]
        # bounds = [ (a-1/len(t)/M, a+1/len(t)/M) for a in normalized_delays[:(k+1)]]
        res = minimize(obj, [normalized_delays[k]], method="Nelder-Mead", bounds=bounds)
        normalized_delays[k] = res.x[0]
        
        Xk = modf.generate_waveform(t, normalized_delays[k]*T*3e8/2)
        for i in range(0, len(residual), window_size):
            amplitude[k,i:i+window_size] = ( np.conj(Xk[i:i+window_size]) @ residual[i:i+window_size] ) / np.sum(np.abs(Xk[i:i+window_size]))
        
        residual = residual - amplitude[k] * Xk
    if get_intermediates:
        return normalized_delays, amplitude, intermediates
    return normalized_delays, amplitude

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


def basis_pursuit(mf, y, lambd, gamma=None, 
                         rho:float=10, iter:int=100, 
                         regularization:Literal["l1", "elastic-net", "mcp"]="l1", tol=1e-5):
    print(regularization)
    if (regularization=="elastic-net") and (gamma is None):
        gamma = lambd / 10
    elif (regularization=="mcp") and (gamma is None):
        gamma = 2
        
    M = int(len(mf)/len(y))
    N = len(y)
    MF = _make_into_subsequences(mf, M)

    b = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    u = np.random.rand(M, N) * np.exp(1j * 2*np.pi* np.random.rand(M, N))
    b /= np.sqrt(np.sum(np.abs(b)**2))
    u /= np.sqrt(np.sum(np.abs(u)**2))

    MF_DFT = fft(MF, axis=-1)
    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # N
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    print("basis pursuit")
    for i_iter in range(iter):
        # a = (XhX)^{-1} @ (conj(X).T @ y + rho * ( b - u ))
        #  Xt@y - Xt@X@a + rho*(b-u) - rho*a 
        # inv(Xt@X + rho) @ (Xt@y + rho*(b-u))
        tmp = mf_freqres + rho * fft(b-u, axis=-1, norm="ortho") # M x N
        tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # M x N
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

def bartlett_basis_pursuit(mf:np.ndarray, y:np.ndarray, 
                                  lambd:float, window_len:int, stride:int, 
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

# def bartlett_low_rank_basis_pursuit(mf:np.ndarray, y:np.ndarray, 
#                                   lambd:float, window_len:int, stride:int,
#                                   group_len:int, group_stride:int,
#                                   rho:float=10, window=None, iter=100):

#     N = len(y)
#     M = int(len(mf)/len(y))

#     MF = _make_segmented_subsequences(mf, M, window_len, stride, window)
#     MF_DFT = fft(MF, axis=-1)

#     L = MF.shape[0]

#     MF_DFT = MF_DFT.reshape((L*M, N)) # L*M x N
    
#     b = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
#     u = np.random.rand(L, M, N) * np.exp(1j * 2*np.pi* np.random.rand(L, M, N))
#     b /= np.sqrt(np.sum(np.abs(b)**2, axis=(1,2), keepdims=True))
#     u /= np.sqrt(np.sum(np.abs(u)**2, axis=(1,2), keepdims=True))

#     diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # 
#     mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    
#     for i_iter in range(iter):
#         tmp = mf_freqres + rho * fft(np.reshape(b, (L*M, N)), axis=-1, norm="ortho") # L*M x N
#         tmp = 1/rho*tmp - 1/rho**2 * np.sum(MF_DFT * tmp, axis=0) / diags * np.conj(MF_DFT) # L*M x N
#         tmp = ifft(tmp, axis=-1, norm="ortho") # L*M x N
#         a = np.reshape(tmp, (L, M*N)) # L x M*N
#         b = np.zeros_like(a)
#         c = np.zeros_like(b)
#         x = np.zeros((M*N,), dtype=a.dtype)
#         for i in range(0, M*N, group_stride):
#             u1, s1, v1 = svds(a[:, i:i+group_len], k=1)
#             x = np.diag(s1)@v1
#             v = np.exp(1j*np.angle(x))*np.maximum(np.abs(x)-lambd, np.zeros_like(x))
#             b[:, i:i+group_len] = u1 @ v
#             c[:, i:i+group_len] += 1
#         b /= c
#         print(np.mean(np.abs(a-b)))
#         print(f"{i_iter+1:d}/{iter:d}")
#     print()
#     a = np.reshape(a, (L, M, N))
#     a = np.transpose(a, (0, 2, 1))
#     a = np.reshape(a, (L, N*M))
#     return a

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

def bartlett_low_rank_basis_pursuit(mf:np.ndarray, y:np.ndarray, 
                                  lambd:float, window_len:int, stride:int,
                                  group_len:int, group_stride:int,
                                  rho:float=10, window=None, iter=100, tol=1e-5):

    N = len(y)
    M = int(len(mf)/len(y))

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
    for i_iter in range(iter):
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