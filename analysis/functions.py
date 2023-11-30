"""functions.py

Miscellaneous functions

"""

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import pycircstat

import scipy as sc


# ------------------ Circular data operations ----------------------- #

def unwrap(signal, period=360):
    unwrapped = np.unwrap(signal * 2 * np.pi / period) * period / np.pi / 2
    return unwrapped


def wrap(arr, cmin=-180, cmax=180):
    period = cmax - cmin
    arr = arr % period
    arr[arr >= cmax] = arr[arr >= cmax] - period
    arr[arr < cmin] = arr[arr < cmin] + period
    return arr


# TODO switch default to 'raise', better practice
def circmean(arr, cmin=-180, cmax=180, nan_policy='omit'):
    return stats.circmean(arr, high=cmin, low=cmax, nan_policy=nan_policy)


def circstd(arr, low=-180, high=180, axis=None, omit_nans=True):
    if omit_nans:
        arr = arr[~np.isnan(arr)]
    return stats.circstd(arr, high, low, axis)


def circ_stderror(arr):
    std = circstd(arr)
    n = len(arr[~np.isnan(arr)])
    return std / np.sqrt(n)


def circcorr(a, b, nan_policy='raise'):
    if nan_policy == 'rase':
        if (np.sum(np.isnan(a)) > 0) | (np.sum(np.isnan(b)) > 0):
            print('contains NaNs')
            return
    elif nan_policy == 'omit':
        idx = ~((np.isnan(a)) | (np.isnan(b)))
        a = a[idx]
        b = b[idx]
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    corr = pycircstat.corrcc(a, b)
    return corr


def circgrad(signal, method=np.gradient, period=360, **kwargs):
    signaluw = unwrap(signal, period)
    dsignal = method(signaluw, **kwargs)
    return dsignal


# Vector manipulation
def cart2polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def circ_interpolate(x, xp, fp):
    # pmp circular interpolation, input has to be in degrees, change variable names!
    a, b = polar2cart(np.ones(len(fp)), np.deg2rad(fp))
    f = interp1d(xp, a, kind='linear', bounds_error=False, fill_value=np.nan)
    a_upsampled = f(x)
    f = interp1d(xp, b, kind='linear', bounds_error=False, fill_value=np.nan)
    b_upsampled = f(x)
    r, theta = cart2polar(a_upsampled, b_upsampled)
    return np.rad2deg(theta)


def mean_vector(theta):
    x, y = polar2cart(np.ones(len(theta)), theta)
    r, theta = cart2polar(x.mean(), y.mean())
    return r, theta


# ------------------ Binary data operations ----------------------- #


def number_islands(x):
    # x binary or boolean sequence
    # numbers islands of ones or True
    # e.g. [0,0,0,1,1,1,0,0,0,1,1,1,1,0,0] --> [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0]
    # might not work if starts with a 1?
    out = np.zeros(len(x), dtype=int)
    starts = np.where(np.diff(x.astype(int)) > 0)[0] + 1
    ends = np.where(np.diff(x.astype(int)) < 0)[0] + 1

    for i_event, (start, end) in enumerate(zip(starts, ends)):
        out[start:end] = i_event + 1

    return out


# ------------------ Statistics ----------------------- #

def mode(a):
    # removes nans, if list is empty it will return nan
    # had to do this bc it was having unexpected behaviour, nan_policy has no effect now
    a = [x for x in a if str(x) != 'nan']
    if len(a) > 0:
        mode = stats.mode(a, nan_policy='omit')[0][0]
    else:
        mode = np.nan
    return mode


def str_mode(a):
    a = list(a)
    max(set(a), key=a.count)


# might not keep
def nanzscore(arr):
    arrout = (arr - np.nanmean(arr)) / np.nanstd(arr)
    return arrout


def xcorr(a, v, sampling_period=1, norm=True):
    """
    Computes normalized cross-correlation between a and v.
    :param a: 1-d vector.
    :param v: 1-d vector of same length as a.
    :param sampling_period: time steps associated with indices in a and v.
    :return: t, delay a_t - v_t (if t>0, a comes after v)
            xc, full cross-correlation between a and v.
    """
    # DOES NOT DEAL WITH NANS PROPERLY PMP

    if not len(a) == len(v):
        print('len(a) must equal len(v).')
        return
    if norm:
        a = nanzscore(a)
        v = nanzscore(v)
    l = len(a)
    a = a / (len(a) - 1)
    xc = np.correlate(a, v, 'full')
    t = np.arange(-(l - 1), (l - 1) + 1) * sampling_period
    return t, xc


# ------------------ Filters ----------------------- #


def butterworth(input_signal, cutoff, passside='low', N=8, sampling_freq=100):
    # Nyquist_freq = sampling_freq / 2.
    # Wn = cutoff/Nyquist_freq
    Wn = cutoff
    b, a = sc.signal.butter(N, Wn, passside, fs=sampling_freq)
    output_signal = sc.signal.filtfilt(b, a, input_signal)
    return output_signal


# ------------------ Miscellaneous ----------------------- #

def contains(list1, list2):
    for el in list1:
        if el in list2:
            return el
    return False
