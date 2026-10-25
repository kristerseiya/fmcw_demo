
import numpy as np
from scipy.signal import correlate


def correlate_partitions(x, y, M, stride, **kwargs):
    rect = np.arange(len(x))
    response = 0
    for i in range(0, len(x), stride):
        response += np.abs(correlate(x*(rect>=i)*(rect<i+M), y, **kwargs))
    return response
        