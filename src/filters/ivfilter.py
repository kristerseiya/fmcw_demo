
import numpy as np
import cvxpy as cp

from .matchfilt import construct_matchfilter
from chirpgen.config import ChirpGeneratorConfig

def compute_minimum_isl_ivfilter(t:np.ndarray, config:ChirpGeneratorConfig, padding=0):
    x = construct_matchfilter(t, config)
    M = padding
    N = len(x)
    xp = np.pad(x, (M, M))
    R = np.zeros((len(xp), 2*M+2*N), dtype=xp.dtype)
    for i in range(R.shape[1]):
        if i < M+N:
            R[i+1:,i] = xp[:-i-1]
        else:
            R[:-(i+1-M-N),i] = xp[(i+1-M-N):]
            
    R = R @ np.conj(R.T) / (2*M+2*N)
    iv = np.linalg.inv(R) @ xp
    iv /= np.abs(xp @ np.conj(iv))
    return iv


def compute_minimum_psl_ivfilter(t:np.ndarray, config:ChirpGeneratorConfig, padding=0):
    x = construct_matchfilter(t, config)
    M = padding
    N = len(x)
    xp = np.pad(x, (M, M))
    Jxp = np.zeros((len(xp), 2*M+2*N), dtype=xp.dtype)
    for i in range(Jxp.shape[1]):
        if i < M+N:
            Jxp[i+1:,i] = xp[:-i-1]
        else:
            Jxp[:-(i+1-M-N),i] = xp[(i+1-M-N):]

    x = cp.Variable(2*(2*M+N)+1)
    A = []
    b = []
    c = np.zeros((2*(2*M+N)+1,))
    c[-1] = 1
    for i in range(Jxp.shape[1]):
        re = np.real(Jxp[:,i])
        im = np.imag(Jxp[:,i])
        Ai = np.vstack([np.hstack([re, im]), np.hstack([-im, re])])
        Ai = np.hstack([Ai, np.zeros((2,1))])
        A.append( Ai )
        tmp = xp @ np.conj(Jxp[:,i])
        b.append( np.array([np.real(tmp), np.imag(tmp)]))
    d1 = np.concatenate([ np.real(xp), np.imag(xp), [0] ])
    d2 = np.concatenate([ -np.imag(xp), np.real(xp), [0] ])
    soc_constraints = [ cp.SOC(c.T@x, A[i]@x+b[i]) for i in range(Jxp.shape[1]) ]
    prob = cp.Problem(cp.Minimize(c.T@x), soc_constraints+[d1.T@x==0]+[d2.T@x==0])
    prob.solve()
    iv = xp + x.value[:(2*M+N)] + 1j * x.value[(2*M+N):-1] 
    iv /= np.abs(xp @ np.conj(iv))
    return iv