__author__ = 'Raymond Chia'
'''
Generic digital signal processing functions.
'''
import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    iirnotch,
    decimate,
    stft,
)
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.fft import fft, fftfreq
from scipy import ndimage

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import emd
from pywt import cwt

from configs import FS

def butter_lowpass(lowcut, fs=FS, order=5):
    ''' Low pass butter filter coefficients '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, analog=False, btype='low', output='ba')
    return b, a

def butter_lowpass_filter(data, lowcut, fs=FS, order=5, axis=-1):
    ''' Apply bidirectional low pass filter '''
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def butter_highpass(highcut, fs=FS, order=5):
    ''' High pass butter filter coefficients '''
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, analog=False, btype='high', output='ba')
    return b, a

def butter_highpass_filter(data, highcut, fs=FS, order=5, axis=-1):
    ''' Apply bidirectional high pass filter '''
    b, a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def butter_bandpass(lowcut, highcut, fs=FS, order=5):
    ''' Band pass butter filter coefficients '''
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], analog=False, btype='band', output='ba')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=FS, order=5, axis=-1):
    ''' Apply bidirectional high pass filter '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def notch_filter(data, notch_freq, Q=30, fs=FS, axis=-1):
    ''' Apply notch filter '''
    b, a = iirnotch(notch_freq, Q, fs)
    y = filtfilt(b, a, data, axis=axis)
    return y

def lin_detrend(data, axis=-1):
    ''' Apply linear detrending '''
    return detrend(data, axis=axis)

def std_scale(data):
    ''' Apply standard scaling '''
    if data.ndim < 2:
        data = data.reshape(-1, 1)

    if data.shape[-1] > data.shape[0]:
        data = data.T

    return StandardScaler().fit_transform(data)

def minmax_scale(data):
    ''' Apply min max scaling '''
    if data.shape[-1] > data.shape[0]:
        data_in = data.T
    else:
        data_in = data.copy()

    return MinMaxScaler().fit_transform(data_in)

def reject_artefact(data, thold=2, percent=0.03):
    ''' Reject values in data window if greater than threshold for some
    percentage of time '''
    N = len(data)
    chk = np.sum(data > thold, axis=0) > N*percent
    if np.any(chk): return True
    else: return False

def downsample(data, q=2, **kwargs):
    ''' Downsample data by a ratio of 'q' '''
    y = decimate(data, q, **kwargs)
    return y

def interpolate(data, fs_resample, fs=FS, **kwargs):
    ''' Interpolate data by a resample rate '''
    N_orig = len(data)
    N_rsam = int(N_orig*fs_resample/fs)

    x_orig = np.linspace(0, 1, N_orig)
    x_rsam = np.linspace(0, 1, N_rsam)

    func = interp1d(x_orig, data, **kwargs)
    y_rsam = func(x_rsam)

    return y_rsam

def run_fft(data, fs=FS):
    ''' Perform Fast Fourier Transform '''
    N = len(data)
    T = 1/fs
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = 2.0/N * np.abs(fft(data)[0:N//2])
    xf = fftfreq(N,T)[:N//2]
    return xf, yf

def do_pad_fft(sig, fs=FS):
    ''' Perform padded Fast Fourier Transform for increased FFT resolution '''
    pad_len = npads_frequency_resolution(len(sig), fs=fs)
    data_pad = np.pad(sig.squeeze(), (0, pad_len), 'constant', constant_values=0)
    data_xf, data_yf = run_fft(data_pad, fs)
    return data_xf, data_yf

def movingaverage(data, window_size, axis=-1, **kwargs):
    ''' Calculate moving average of data for some window size '''
    data_in = data.copy()
    return uniform_filter1d(
        data_in,size=window_size, mode='constant',
        axis=axis, **kwargs
    )

def run_cwt(data, scales=None, wavelet='morl', **kwargs):
    ''' Perform continuous wavelet transform '''
    if scales is None:
        scales = np.arange(1, 129)

    return cwt(data, scales, wavelet, **kwargs)

def run_stft(data, fs=FS, kernel_size=32, stride_size=8):
    ''' Perform Short-time Fourier transform '''
    f, t, z = stft(data, fs=fs, nperseg=kernel_size,
                   noverlap=kernel_size-stride_size, padded=True)
    return f, t, z


def npads_frequency_resolution(data_len, fr=0.02, fs=FS):
    nbins = fs//fr
    npads = nbins*2 - data_len
    return int(npads)

def run_hht(data, freq_range, fs=FS, mode='amplitude', sum_time=False, **kwargs):
    ''' Perform Hilbert-Huang transform '''
    imf = emd.sift.sift(data)
    IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'hilbert')
    hht_f, hht = emd.spectra.hilberthuang(IF, IA, freq_range,
                                          scaling='density', mode=mode,
                                          sum_time=sum_time, **kwargs)
    hht = ndimage.gaussian_filter(hht, 1)
    return hht_f, hht
