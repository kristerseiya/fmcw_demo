
import numpy as np
from scipy.sparse.linalg import lsqr, factorized
import scipy.sparse as sparse
import tf_lib.chirplet_ridge as CR
from tf_lib import if_lib
import scipy as sp
import scipy.sparse as spsp
import scipy.linalg as spla
from typing import Literal, List
from scipy.fft import fftfreq, fft, ifft
import scipy.integrate as spint

class PicklableFactorized:
    """
    Picklable version of scipy.sparse.linalg.factorized(M): solver(b) solves M x = b.

    The SuperLU object behind factorized() cannot be pickled, so it cannot be sent
    to worker processes. This only pickles M and redoes the factorization when
    unpickled (i.e. once per worker).
    """
    def __init__(self, M):
        self.M = sparse.csc_matrix(M)
        self._solve = factorized(self.M)

    def __call__(self, b):
        return self._solve(b)

    def __getstate__(self):
        return {"M": self.M}

    def __setstate__(self, state):
        self.__init__(state["M"])


def construct_fractional_D_with_time(t_vector, tau, method='linear', P=5):
    """
    Constructs the difference matrix D for a non-integer shift tau and 
    returns the timestamps corresponding to each row of D.
    
    Parameters:
        t_vector (array): Complete 1D array of timestamps for the N samples.
        tau (float): The shift amount in units of time (same units as t_vector).
        method (str): 'linear' or 'sinc'.
        P (int): Half-width of the sinc window (only used if method='sinc').
        b
    Returns:
        D (csr_matrix): The design matrix.
        t_observed (array): Timestamps corresponding row-by-row to D.
    """
    N = len(t_vector)
    dt = t_vector[1] - t_vector[0]  # Assuming uniform sampling
    
    # Convert tau from time units to sample units
    tau_samples = tau / dt
    k = int(np.floor(tau_samples))
    delta = tau_samples - k
    
    if method == 'linear':
        start_idx = k + 1
        end_idx = N
        M = end_idx - start_idx
        
        rows = np.repeat(np.arange(M), 3)
        cols = np.zeros(3 * M, dtype=int)
        data = np.zeros(3 * M)
        
        idx = 0
        for i in range(start_idx, end_idx):
            row_in_D = i - start_idx
            
            # f(t)
            cols[idx] = i
            data[idx] = 1.0
            idx += 1
            
            # -(1 - delta) * f(t - k)
            cols[idx] = i - k
            data[idx] = -(1.0 - delta)
            idx += 1
            
            # -delta * f(t - k - 1)
            cols[idx] = i - k - 1
            data[idx] = -delta
            idx += 1
            
        D = sparse.csr_matrix((data, (rows, cols)), shape=(M, N))
        
    elif method == 'sinc':
        start_idx = k + P
        end_idx = N - P
        M = end_idx - start_idx
        
        num_elements_per_row = 1 + (2 * P + 1)
        rows = np.repeat(np.arange(M), num_elements_per_row)
        cols = np.zeros(num_elements_per_row * M, dtype=int)
        data = np.zeros(num_elements_per_row * M)
        
        idx = 0
        for i in range(start_idx, end_idx):
            # f(t) component
            cols[idx] = i
            data[idx] = 1.0
            idx += 1
            
            # f(t - tau) component
            j_range = np.arange(i - k - P, i - k + P + 1)
            x = (i - tau_samples) - j_range
            
            weights = np.sinc(x) * np.sinc(x / P)
            weights /= np.sum(weights)
            
            for j, w in zip(j_range, weights):
                cols[idx] = j
                data[idx] = -w
                idx += 1
                
        D = sparse.csr_matrix((data, (rows, cols)), shape=(M, N))
        
    # Extract the matching timestamps for the rows
    t_observed = t_vector[start_idx:end_idx]
    
    return D, t_observed


def construct_diff_matrix(N):
    x = np.ones((2*N)) / 2
    x[:N] = -1/2
    rows = np.concatenate([np.arange(N)]*2)
    cols = np.concatenate([np.arange(N), np.arange(2,N+2)])
    D = sparse.csc_matrix((x, (rows, cols)), shape=(N, N+2))
    return D


def estimate_if_fcrm(signal, **kwargs):
    kwargs.setdefault("fs", 1000)
    kwargs.setdefault("sigma", 8.0)
    kwargs.setdefault("nfft", 128)
    kwargs.setdefault("n_beta_in", 16)
    kwargs.setdefault("n_omega_out", 256)
    kwargs.setdefault("n_beta_out", 48)
    kwargs.setdefault("hop", 2 * max(1, int(round(kwargs["fs"] / 250))))
    kwargs.setdefault("beta_max_hz_s", kwargs["fs"]**2 / (8 * kwargs["sigma"]) )
    kwargs.setdefault("max_dk", int(np.ceil(kwargs["beta_max_hz_s"] * kwargs["hop"] / kwargs["fs"]**2 * kwargs["n_omega_out"] + 3)))
    kwargs.setdefault("max_db", max(2, kwargs["n_beta_out"]//4))
    kwargs.setdefault("lam_k", 0.1)
    kwargs.setdefault("lam_b", 0.5)
    # fs = 1000 
    # sigma, K, Bn = 8.0, 256, 48
    # hop = 2 * max(1, int(round(fs / 250)))          # FCRM hop in analyze_data
    # max_chirp = fs**2 / (8 * sigma)                 # chirp-rate range ±Hz/s
    # df = fs / K                                     # two-sided bin width (Hz)
    # max_jump = max_chirp * hop / fs + 3 * df

    # ref_if_t, ref_if_est, _ = CR.estimate_ifs(
    #     signal, fs, n_components=1, sigma=sigma, hop=hop, nfft=128, n_beta_in=16,
    #     beta_max_hz_s=max_chirp, n_omega_out=K, n_beta_out=Bn, mode="eq9", onesided=False,
    #     ridge="dp", lam_k=lam_k, lam_b=lam_b, max_dk=int(np.ceil(max_jump / df)),
    #     max_db=max(2, Bn // 4), prediction=True)
    if_t, if_est, _ = CR.estimate_ifs(
            signal, n_components=1, mode="eq9", onesided=False,
            ridge="dp", prediction=True, **kwargs)
    if_t = if_t*kwargs["fs"]
    if_est = if_est[0]/kwargs["fs"]
    return if_t, if_est

def estimate_if_hilbert(signal, **kwargs):
    kwargs.setdefault("fs", 1000)
    kwargs.setdefault("filt", None)
    kwargs.setdefault("tp_params", dict(f_lo=20))

    if_t, if_est, _ = if_lib.signed_hilbert_estimate(signal, **kwargs)
    if_est = if_est / kwargs["fs"]
    if_t = if_t * kwargs["fs"]

    return if_t, if_est

class ModuFuncSolver():
    def __init__(self, N:int, sampling_rate:float, delays:List[float], period=None, 
                 method:Literal["ridge", "pinv", "fft"]="ridge", verbose=False):

        self.method = method
        self.period = period

        if self.method == "ridge":
            if verbose:
                print("computing factorized linear operator... ", end="")
            self.ts_obs = []
            As = []
            for delay in delays:
                A, t_obs = construct_fractional_D_with_time(np.arange(N)/sampling_rate, delay, method="linear", P=10)
                As.append(A)
                self.ts_obs.append(t_obs)
            A = spsp.vstack(As)
            I = spsp.eye(A.shape[1])
            AtAI = A.T @ A + 1e-7 * I
            self.At = A.T
            self.ridge_solver = PicklableFactorized(AtAI)
            if verbose:
                print("done")
            
        elif self.method == "pinv":

            if verbose:
                print("computing pseduo inverse... ", end="")
            self.ts_obs = []
            As = []
            for delay in delays:
                A, t_obs = construct_fractional_D_with_time(np.arange(N)/sampling_rate, delay, method="linear", P=10)
                As.append(A)
                self.ts_obs.append(t_obs)
            A = spsp.vstack(As)
            self.Apinv = spla.pinv(A.toarray())
            if verbose:
                print("done")

        elif self.method == "fft":

            if verbose:
                print("computing fft inverse... ", end="")
            omega = 2*np.pi*fftfreq(N)
            F = (np.abs(omega)<(2*np.pi*0.01))*1.0 + (np.abs(omega)>=(2*np.pi*0.01))*1e-3
            As = []
            diag = 0
            for delay in delays:
                A = 1-np.exp(1j*omega*delay*sampling_rate)
                As.append(A)
                diag += np.abs(A)**2
            self.fftinv = np.conj(np.stack(As))/(diag + 1e-6*F)
            self.t_obs = np.arange(N)/sampling_rate
            if verbose:
                print("done")

        else:
            raise ValueError(f"did not recognized {self.method:s}")
        
    def estimate(self, *ifqs):

        if self.method == "ridge":
            ifq_resamps = []
            for (ift, ifq), t_obs in zip(ifqs, self.ts_obs):
                ifq_resamp = np.interp(t_obs, ift, ifq, period=self.period)
                ifq_resamps.append(ifq_resamp)
            ifq_resamps = np.concatenate(ifq_resamps)
            a_est = self.ridge_solver(self.At@ifq_resamps)

        elif self.method == "pinv":
            ifq_resamps = []
            for (ift, ifq), t_obs in zip(ifqs, self.ts_obs):
                ifq_resamp = np.interp(t_obs, ift, ifq, period=self.period)
                ifq_resamps.append(ifq_resamp)
            ifq_resamps = np.concatenate(ifq_resamps)      
            a_est = self.Apinv @ ifq_resamps

        elif self.method == "fft":
            ifq_resamps = []
            for (ift, ifq) in ifqs:
                ifq_resamp = np.interp(self.t_obs, ift, ifq, period=self.period)
                ifq_resamps.append(ifq_resamp)
            ifq_resamps = np.stack(ifq_resamps)
            a_est = ifft( np.sum(self.fftinv*fft(ifq_resamp, norm="ortho"), axis=0), norm="ortho")
            a_est = np.real(a_est)

        else:
            raise RuntimeError()

        return a_est

def subtract_signal(t, y, ift, ifq, window_len, period=None):
    ifq_resamp = np.interp(t, ift, ifq, period=period)
    phase = spint.cumulative_simpson(ifq_resamp, x=t, initial=0)
    main_signal = np.exp(1j*2*np.pi*phase)
    subtracted_y = np.zeros_like(y)
    for i in range(0, len(t), window_len):
        X = np.stack([ main_signal[i:i+window_len], np.conj(main_signal[i:i+window_len]) ])
        alpha = np.linalg.inv(X @ np.conj(X).T) @ np.conj(X) @ y[i:i+window_len]
        subtracted_y[i:i+window_len] = y[i:i+window_len] - np.real(alpha@X)
    return subtracted_y