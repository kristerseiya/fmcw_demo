
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

def _make_into_subsequences(x, M):
    tmp = np.reshape(np.roll(x, M-1), (-1, M))
    tmp = np.transpose(np.flip(tmp, axis=1))
    return tmp


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
    diags = 1 + 1/rho*np.sum(np.square(np.abs(MF_DFT)), axis=0) # N
    mf_freqres = fft(y,norm="ortho")*np.conj(MF_DFT)
    a_old = np.zeros_like(b)
    print("basis pursuit")
    for i_iter in range(n_iter):
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