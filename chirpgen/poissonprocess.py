
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from typing import Union

class IntegralApproximator():
    def __init__(self, 
                 func: np.ndarray, 
                 bin_edges: np.ndarray):
        
        # check if bin edges are sorted
        assert( all(bin_edges[i] <= bin_edges[i + 1] for i in range(len(bin_edges) - 1)) )

        self.func = func
        self.bin_edges = bin_edges
        self.eval_points = 0.5*(bin_edges[1:] + bin_edges[:-1])

        if isinstance(func, Callable):

            # self.f_evals = self.func(self.eval_points)
            # self.f_evals /= (self.bin_edges[1:]-self.bin_edges[:-1])
            raise NotImplementedError()

        elif isinstance(func, np.ndarray):

            self.f_evals = self.func * (self.bin_edges[1:]-self.bin_edges[:-1])

        self.f_evals_cumsum = np.cumsum(self.f_evals)

    def integrate(self, lower, upper):

        assert( lower >= self.bin_edges[0] )
        assert( upper <= self.bin_edges[-1] )

        lower_idx = np.searchsorted(self.bin_edges, lower, side='right') - 1
        upper_idx = np.searchsorted(self.bin_edges, upper, side='right')

        return np.sum(self.f_evals[lower_idx:upper_idx])
    
    def inverse(self, lower, a):

        assert( lower >= self.bin_edges[0] )

        lower_idx = np.searchsorted(self.bin_edges, lower, side='right') - 1

        tmp = self.f_evals_cumsum[lower_idx:] - self.f_evals_cumsum[lower_idx]

        idx = np.searchsorted(tmp, a, side='right') 

        if lower_idx + idx >= len(self.bin_edges):
            return self.bin_edges[-1]
            
        return self.bin_edges[lower_idx+idx]
        

class PoissonProcess():
    def __init__(self, rate_function, bin_edges, period):

        self.period = period
        self.int_approx = IntegralApproximator(rate_function, bin_edges)

    def generate(self):

        lower = 0

        time_stamps = []

        U = np.random.rand()
        a = - np.log(1-U)
        lower = self.int_approx.inverse(lower, a)

        while lower < self.period:

            time_stamps.append(lower)
            U = np.random.rand()
            a = - np.log(1-U)
            lower = self.int_approx.inverse(lower, a)

        return np.array(time_stamps)
