__author__ = 'Raymond Chia'
"""
Feature calculation functions. Only non-self-explanatory functions
have documentation
"""

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis, entropy
from EntropyHub import SampEn, ApEn

from warnings import warn
from itertools import combinations

from dsp import run_fft
from utils import get_windows

from configs import FS

from inspect import getmembers, isfunction, currentframe

def return_all_features(data, ch=0, **kwargs):
    '''
    Returns all feature calculations in this file for a singular channel

    :param data: Input data for feature calculation. This is flattened before
    calculations
    :type type: numpy.ndarray or numpy like array
    :param ch: Input channel
    :type ch: int

    :return: Dictionary of features from input data, keys are prefixed by
    channel integer
    :return type: dict
    '''
    if data.ndim > 1:
        data = data.flatten()
    global_dict = currentframe().f_globals
    funcs = {key: val for key, val in global_dict.items() if 'get_' in key}
    out_dict = {}

    def append_to_key(in_dict, val):
        for i, val in enumerate(tmp):
            in_dict[key_name+f'_{i}'] = val

    for func_name, func in funcs.items():
        # skip these functions
        if func_name in ['get_windows', 'get_peaks']: continue

        key_name = f'ch{ch}_' + func_name[4:]
        if func_name == 'get_psd':
            freqs, psd = func(data, **kwargs)
        elif 'psd' in func_name or 'power_band' in func_name:
            tmp = func(freqs, psd, **kwargs)
            append_to_key(out_dict, tmp)
        else:
            tmp = func(data, **kwargs)
            if isinstance(tmp, list) and len(tmp) > 1:
                append_to_key(out_dict, tmp)
            else:
                out_dict[key_name] = tmp
    return out_dict

def get_mean(data, axis=-1, **kwargs):
    return np.mean(data, axis=axis, **kwargs)

def get_std(data, axis=-1, **kwargs):
    return np.std(data, axis=axis, **kwargs)

def get_median(data, axis=-1, **kwargs):
    return np.median(data, axis=axis, **kwargs)

def get_max(data, axis=-1, **kwargs):
    return np.max(data, axis=axis, **kwargs)

def get_min(data, axis=-1, **kwargs):
    return np.min(data, axis=axis, **kwargs)

def get_pk2pk(data, axis=-1, **kwargs):
    return np.ptp(data, axis=axis, **kwargs)

def get_skew(data, axis=-1, **kwargs):
    return skew(data, axis=axis, **kwargs)

def get_kurtosis(data, axis=-1, **kwargs):
    return kurtosis(data, axis=axis, **kwargs)

def get_sample_entropy(data, m=3, r=0.25, **kwargs):
    '''
    Returns the mean sample entropy of the 1-D array. Can be considered 
    equivalent to approximate entropy with computation differences. See
    <https://www.entropyhub.xyz/index.html> for documentation.

    :param data: 1-D array for processing
    :type data: numpy.ndarray or numpy like array
    :param m: number of sub-windows to develop distributions
    :type m: int
    :param r: Tolerance, distance between points
    :type r: float

    :return: sample entropy
    :return type: float
    '''
    if data.ndim > 1:
        warn("SampEn requires 1-D array. Input data is flattened",
             RuntimeWarning)
        data = data.flatten()
    se, _, _ = SampEn(data, m=m, r=r)
    return np.mean(se)

def get_approx_entropy(data, m=3, r=0.25, **kwargs):
    '''
    Returns the mean approximate entropy of the 1-D array. Quantifies 
    irregularity or fluctuation over time-series data. See
    <https://www.entropyhub.xyz/index.html> for documentation.

    :param data: 1-D array for processing
    :type data: numpy.ndarray or numpy like array
    :param m: number of sub-windows to develop distributions
    :type m: int
    :param r: Tolerance, distance between points
    :type r: float

    :return: approximate entropy
    :return type: float
    '''
    if data.ndim > 1:
        warn("ApEn requires 1-D array. Input data is flattened",
             RuntimeWarning)
        data = data.flatten()
    ap, _ = ApEn(data, m=m, r=r)
    return np.mean(ap)

def get_ieeg(data, axis=-1, **kwargs):
    '''
    Returns the area under curve for the input data.

    :param data: array for processing
    :type data: numpy.ndarray or numpy like array
    :param axis: axis to operate on
    :type axis: int

    :return: integrated EEG
    :return type: float
    '''
    return np.trapz(np.abs(data), axis=axis, **kwargs)

def get_rms(data, axis=-1, **kwargs):
    '''
    Returns the root mean square value for the input data.

    :param data: array for processing
    :type data: numpy.ndarray or numpy like array
    :param axis: axis to operate on
    :type axis: int

    :return: root mean square
    :return type: float
    '''
    return np.sqrt(np.mean(data**2, axis=axis, **kwargs))

def get_max_freq(data, fs=FS):
    '''
    Returns the maximum frequency of the multi-channel data after performing
    Fast Fourier Transform.

    :param data: Array for processing
    :type data: numpy.ndarray or numpy like array
    :param fs: sample frequency
    :type fs: float

    :return: Maximum frequency
    :return type: float
    '''
    if data.ndim < 2:
        data = data.reshape(-1, 1)

    if data.shape[0] < data.shape[1]:
        data = data.T

    nchannels = data.shape[-1]

    fmax = []
    for ch in range(nchannels):
        data_in = data[:, ch]
        xf, yf = run_fft(data_in, fs)

        # max frequency
        fmax.append(xf[yf.argmax()])

    if len(fmax) == 1:
        return fmax.pop()
    else:
        return np.array(fmax)

def get_psd(data, fs=FS, nperseg=10):
    '''
    Returns the power spectral density of the 1-D data after performing
    Welch's method of dividing data into overlapping segments and computing and
    averaging periodogram for each segment.

    :param data: 1-D array for processing
    :type data: numpy.ndarray or numpy like array
    :param fs: sample frequency
    :type fs: float
    :param nperseg: number of samples per segment
    :type nperseg: int

    :return: frequency range and power spectral density
    :return type: tuple
    '''
    f, psd = welch(data, fs, nperseg=nperseg)
    return f, psd

def get_mean_power_band(f, psd, fs=FS, bands=[0, 12, 30, -1]):
    '''
    Returns the average power of the specified frequency bands from the
    outputs of 'get_psd'. Frequency bands are specified as the consecutive
    interval between elements of the 'bands' parameter.

    :param f: frequency output from get_psd
    :type f: numpy.ndarray or numpy like array
    :param psd: complex output from get_psd
    :type psd: numpy.ndarray or numpy like array
    :param bands: consecutive frequency bands to calculate averages
    :type bands: list

    :return: average power in frequency bands
    :return type: numpy.ndarray
    '''
    mean_psd = []
    for i, fr in enumerate(bands[:-1]):

        if bands[i+1] == -1:
            bands[i+1] = fs//2

        mask = (f >= fr) & (f <= bands[i+1])
        mean_psd.append(np.mean(np.abs(psd[mask])))

    return np.array(mean_psd)

def get_se_power_band(f, psd, fs=FS, bands=[0, 12, 30, -1]):
    '''
    Returns the spectral entropy of the specified frequency bands from the
    outputs of 'get_psd'. Frequency bands are specified as the consecutive
    interval between elements of the 'bands' parameter.

    :param f: frequency output from get_psd
    :type f: numpy.ndarray or numpy like array
    :param psd: complex output from get_psd
    :type psd: numpy.ndarray or numpy like array
    :param bands: consecutive frequency bands to calculate averages
    :type bands: list

    :return: spectral entropy
    :return type: numpy.ndarray
    '''
    se_psd = []
    for i, fr in enumerate(bands[:-1]):

        if bands[i+1] == -1:
            bands[i+1] = fs//2

        mask = (f >= fr) & (f <= bands[i+1])
        se_psd.append(entropy(psd[mask]))

    return np.array(se_psd)

def get_n_zero_crossings(data):
    '''
    Returns the number of zero crossings of the 1-D input array.

    :param data: 1-D array to calculate zero crossings
    :type data: numpy.ndarray or numpy like array

    :return: number of zero crossings
    :return type: int
    '''
    if data.ndim < 2:
        data = data.reshape(1, -1)
    return len(np.where(np.diff(np.signbit(data)))[1])

def get_peaks(data, distance=100, height=0.01, **kwargs):
    '''
    Returns the indices and properties of the peaks from the 1-D input data. 
    Refer to scipy documentation.

    :param data: 1-D array to find peaks
    :type data: numpy.ndarray or numpy like array

    :return: peak indices and peak properties in dict form
    :return type: tuple
    '''
    if data.ndim > 1:
        warn("find_peaks requires 1-D array. Input data is flattened",
             RuntimeWarning)
        data = data.flatten()
    return find_peaks(data, distance=distance, height=height, **kwargs)

def get_npks(data, **kwargs):
    '''
    Returns the number of peaks from get_peaks calculation.

    :param data: 1-D array to find peaks
    :type data: numpy.ndarray or numpy like array

    :return: number of peaks
    :return type: int
    '''
    idxs, _ = get_peaks(data, **kwargs)
    return len(idxs)
