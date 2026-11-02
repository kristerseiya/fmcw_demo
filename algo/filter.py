
import numpy as np
from filters import SignalModel
from utils import upsample


from scipy.fft import ifft, fft
from scipy.signal import find_peaks, get_window

from typing import List

def matchedfilter(signal_model:SignalModel, 
                  t:np.ndarray, y:np.ndarray, 
                  uprate:int, t_low:float=None, t_high:float=None, top_k:int=10,
                  **find_peaks_params):
    M = uprate
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)
    y_up = upsample(y, M)
    mf_output = np.abs(ifft(fft(y_up,norm="ortho")*np.conj(fft(mf)),norm="ortho"))**2 / len(y)
    search_region = np.ones((len(t_fine),), dtype=bool)
    if t_low:
        search_region *= (t_fine>=t_low)
    if t_high:
        search_region *= (t_fine<=t_high)
    peaks, _ = find_peaks(mf_output[search_region], **find_peaks_params)
    peaks = peaks + np.argmax(search_region)
    a = np.argsort(mf_output[peaks])[-top_k:]
    d = t_fine[peaks][a]
    p = mf_output[peaks][a]
    return (d, p), (t_fine[search_region], mf_output[search_region])

def _get_start_end_of_each_window(N, L, s):
    n_win = (N - L) // s + 1
    starts = np.arange(n_win) * s
    ends = np.concatenate([ starts[:-1]+L, [N,] ])
    return starts, ends
    
def bartlett_matchedfilter(signal_model:SignalModel, 
                                t:np.ndarray, y:np.ndarray, 
                                uprate:int, 
                                window_len:int,
                                stride:int=None,
                                window:str="rect",
                                t_low:float=None, t_high:float=None, top_k:int=10,
                                **find_peaks_params):
    if stride is None:
        stride = window_len // 2
    assert(window_len >= stride)

    M = uprate
    del_t = t[1]-t[0]
    t_fine = np.linspace(t[0],t[-1]+(M-1)/M*del_t, M*len(t))
    mf = signal_model.get_complex_waveform(t_fine)

    y_up = upsample(y, M)
    mf_output = np.zeros((len(y_up),))
    sidx, eidx = _get_start_end_of_each_window(len(y)*M, window_len*M, stride*M)
    W = get_window(window, window_len*M)
    for si, ei in zip(sidx, eidx):
        rect = np.zeros_like(y_up)
        if (ei-si) == window_len*M:
            rect[si:ei] = W
        else:
            rect[si:ei] = get_window(window, (ei-si))
        mf_output += np.abs(ifft(fft(y_up*rect,norm="ortho")*np.conj(fft(mf)),norm="ortho"))**2 * np.ceil((ei-si)/M)
    mf_output /= len(y)

    search_region = np.ones((len(t_fine),), dtype=bool)
    if t_low:
        search_region *= (t_fine>=t_low)
    if t_high:
        search_region *= (t_fine<=t_high)
    peaks, _ = find_peaks(mf_output[search_region], **find_peaks_params)
    peaks = peaks + np.argmax(search_region)
    a = np.argsort(mf_output[peaks])[-top_k:]
    d = t_fine[peaks][a]
    p = mf_output[peaks][a]
    return (d, p), (t_fine[search_region], mf_output[search_region])
