"""read_data.py

Module for reading ABF files and TIFF images

"""
import glob
import os
from collections import namedtuple
import re
from neo.io import AxonIO
from neo.io import WinEdrIO
import pyabf
import pandas as pd
import numpy as np
import skimage.external.tifffile as tifffile
import joblib
from sklearn.cluster import KMeans
import scipy as sc
from scipy.io import loadmat
from scipy.interpolate import interp1d
import cv2
from shapely.geometry import Point, Polygon

import functions as fc


# ------------------ 1. Behaviour data ----------------------- #

class ABF(object):
    """
        Holds data from ABF file

       Reads in an abf file and stores it as a pandas dataframe.

       :param filename: string specifying path

       :returns Pandas DataFrame


    """

    def __init__(self, filename, neo=True):
        self.filename = filename
        self.rec_name = filename.split('/')[-1].split('.')[0]
        self.start_time = None
        print("loading file: " + filename)

        if neo:
            abf = self.read_abf()

            self.df_orig = pd.DataFrame(columns=abf.keys())

            example_signal = abf[list(abf.keys())[0]]

            # gets sampling rate and period of abf data
            self.sampling_period = float(example_signal.sampling_period)
            self.sampling_rate = float(example_signal.sampling_rate)

            for key in abf:
                self.df_orig[key] = np.array(abf[key]).flatten()
            # adds time as a column
            t_orig = np.array(example_signal.times, dtype=np.float32)
            self.df_orig.insert(0, 't', t_orig)

        else:
            abf = self.read_abf_pyabf()
            self.df_orig = pd.DataFrame(columns=abf.keys())
            for key in abf:
                self.df_orig[key] = np.array(abf[key]).flatten()

    def read_abf(self):
        """
        Returns a dictionary. Keys are the channel names in lowercase, values are the channels.
        """
        extension = self.filename.split('.')[-1]
        if extension == 'EDR':
            io = WinEdrIO
        elif extension == 'abf':
            io = AxonIO
        else:
            raise ValueError('Extension not recognized.')

        analog_signals_dict = {}
        if io == AxonIO:
            fh = io(filename=self.filename)
            segments = fh.read_block().segments

            if len(segments) > 1:
                print('More than one segment in file.')
                return 0

            analog_signals_ls = segments[0].analogsignals
            for analog_signal in analog_signals_ls:
                analog_signals_dict[analog_signal.name.lower()] = analog_signal

            # stores computer time of start of recording
            t = fh.raw_annotations['blocks'][0]['rec_datetime']
            self.start_time = t

        if io == WinEdrIO:
            fh = WinEdrIO(filename=self.filename)
            segments = fh.read_segment(lazy=False)

            for analog_signal in segments.analogsignals:
                analog_signals_dict[analog_signal.name.lower()] = analog_signal

        return analog_signals_dict

    def read_abf_pyabf(self):
        abf = pyabf.ABF(self.filename)
        # not sure if this is same as neo?
        self.start_time = abf.abfDateTime
        self.sampling_rate = float(abf.sampleRate)
        self.sampling_period = 1 / self.sampling_rate
        abf_dict = {}
        abf_dict['t'] = abf.sweepX
        for i, channel in enumerate(abf.adcNames):
            abf_dict[channel] = abf.data[i, :]
        return abf_dict


class Walk(ABF):
    """
        Child class of ABF for ball walking data

        This class will process the abf dataframe (df_orig) and store certain channels in a new data frame (df):

        -downsample certain channels (based on camera trgger or user specified subsampling_rate)
        -convert volts to degrees for xstim, forw, head and side
        -shifts xstim,forw,head and side  by delay_ms*
        -calculates dside, dforw...
        -calcualtes x and y for trajectory
        -rounds stimid -> feature under construction

        :param camtrig_label: channel used for downsampling
        :param subsampling_rate: user specified downsampling rate
        (only needs to be specified if camtrig_label is None)
        :param ball_diam_mm: ball diameter (mm) :param
        arena_edge: edge of arena (deg)
        :param delay_ms: delay between camera trigger and FicTrac update (ms)
        :param angle_offset: offset between xstim and front of arena (deg)
        :param maxvolts: maximum voltage of behaviour channels
        :param stimid_map: dictionary with stimid values as keys and their meaning (e.g. "Dark", "OL",
        "CL"...) as values
        :param matlab_signals: boolean, stores upsampled temperature saved in MATLAB file**

        *in practice this is calculated by turning on/off the camera triggers and seeing how much time elapses
        between last trigger and the next change in heading/side/forw

        **uses unix time, there is ~40 ms/hour drift relative to axoscope time


    """

    def __init__(self, folder, filename, camtrig_label='camtrig', subsampling_rate=None,
                 ball_diam_mm=7.9375, arena_edge=135, delay_ms=30, angle_offset=86, maxvolts=10, stimid_map=None,
                 matlab_signals=False, boxcar_average=None, neo=True):
        ABF.__init__(self, filename, neo)
        self.folder = folder
        self.camtrig_label = camtrig_label
        self.subsampling_rate = subsampling_rate
        self.ball_diam_mm = ball_diam_mm
        self.arena_edge = arena_edge
        self.delay_ms = delay_ms
        self.angle_offset = angle_offset
        self.maxvolts = maxvolts
        self.stimid_map = stimid_map
        self.boxcar_average = boxcar_average

        # renames columns if necessary
        camtrig_labels = fc.contains(['camtrig', 'walkcamtr', 'WalkCamTr'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={camtrig_labels: "camtrig"})

        head_labels = fc.contains(['heading', 'head', 'ball_head'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={head_labels: "heading"})

        forw_labels = fc.contains(['forw', 'ball_forw'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={forw_labels: "forw"})

        side_labels = fc.contains(['side', 'ball_side'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={side_labels: "side"})

        xstim_labels = fc.contains(['xstim', 'stim_x', 'STIM_X'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={xstim_labels: "xstim"})

        ystim_labels = fc.contains(['ystim', 'stim_y', 'STIM_Y'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={ystim_labels: "ystim"})

        wstim_labels = fc.contains(['wstim', 'stim_w'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={wstim_labels: "wstim"})

        ipatch_labels = fc.contains(['patch_1', 'Ipatch'], self.df_orig.columns)
        self.df_orig = self.df_orig.rename(columns={ipatch_labels: "patch_1"})

        # shifts ball channels by delay_ms, this adds NaN values to the last rows
        shift_n_rows = -1 * int((self.delay_ms / 1000.) * self.sampling_rate)
        self.df_orig['heading'] = self.df_orig['heading'].shift(periods=shift_n_rows)
        self.df_orig['side'] = self.df_orig['side'].shift(periods=shift_n_rows)
        self.df_orig['forw'] = self.df_orig['forw'].shift(periods=shift_n_rows)

        # downsample channels based on camera triggers or user specified subsampling rate and stores them
        # in a new data frame

        downsampled_channels = ['t', 'heading', 'side', 'forw', 'puffer', 'xstim', 'ystim', 'stimid', 'temp',
                                'pockels', 'patch_1', 'patch_2', 'meta', 'flow', 'in7', 'wstim', 'servopos']
        downsampled_channels = [channel for channel in downsampled_channels if
                                channel in self.df_orig.columns.to_list()]

        self.df = pd.DataFrame(columns=downsampled_channels)
        subsampling_inds = self.get_subsampled_indices()
        for downsampled_channel in downsampled_channels:
            self.df[downsampled_channel] = self.df_orig[downsampled_channel][subsampling_inds]

        # converts behaviour_channels from volts to degrees and calculates dforw, dside and dhead
        behaviour_channels = ['heading', 'side', 'forw', 'xstim', 'wstim', 'servopos']
        for behaviour_channel in behaviour_channels:
            if behaviour_channel in self.df.columns:
                if behaviour_channel == 'servopos':
                    self.df[behaviour_channel] = fc.wrap(self.volts2degs(self.df[behaviour_channel], 5))
                elif behaviour_channel == 'xstim':
                    self.df[behaviour_channel] = fc.wrap(self.volts2degs(self.df[behaviour_channel], 10))
                else:
                    self.df[behaviour_channel] = fc.wrap(self.volts2degs(self.df[behaviour_channel], self.maxvolts))
                self.df["d" + behaviour_channel] = fc.circgrad(self.df[behaviour_channel]).astype(
                    np.float32) * self.subsampling_rate

        # converts to mm/s
        self.df['dforw'] = self.deg2mm(self.df['dforw']).astype(np.float32)
        self.df['dside'] = self.deg2mm(self.df['dside']).astype(np.float32)

        # compensates for angular offset between pattern and voltage to angular mapping for xstim
        self.df['xstim'] = fc.wrap(self.df['xstim'] + self.angle_offset)

        # calculates X Y trajectory, uses bar position instead of heading since bar is sometimes in open-loop
        # xstim needs to be *-1 (since heading and xstim are mirrored) and X and Y need to be swapped
        # to make it so that up is towards bar
        x = []
        y = []
        x0, y0 = 0, 0
        # need to convert dside and dforw to mm/frame
        dside = self.df['dside'] / self.subsampling_rate
        dforw = self.df['dforw'] / self.subsampling_rate
        for head, dside, dforw in zip(np.deg2rad(-1 * self.df['xstim']), dside, dforw):
            dx = (dforw * np.cos(head)) + (dside * np.sin(head))
            dy = (dforw * np.sin(head)) - (dside * np.cos(head))
            x.append(x0 + dx)
            y.append(y0 + dy)
            x0, y0 = x[-1], y[-1]
        self.df['x'] = np.array(y).astype(np.float32)
        self.df['y'] = np.array(x).astype(np.float32)

        if matlab_signals:
            self.get_matlab_signals()

    def get_subsampled_indices(self):
        """
           Returns  indices for subsampled data based on camera trigger or user specifies sub_sampling rate
        """
        if self.camtrig_label:
            # gets indices of camera triggers
            subsampling_inds = get_trig(self.df_orig[self.camtrig_label])
            # store subsampling rate (i.e. camera rate)
            self.subsampling_rate = np.float32(self.sampling_rate / np.diff(subsampling_inds).mean())

        else:
            # if no camera trigger then subsamples without triggers with user specified subsampling rate,
            subsampling_step = int(self.sampling_rate / self.subsampling_rate)
            inds = np.arange(self.df_orig['t'].size)
            subsampling_inds = inds[::subsampling_step]

        return subsampling_inds

    def volts2degs(self, volts, maxvolts):
        degs = volts * 360 / maxvolts
        degs[degs > 360] = 360.
        degs[degs < 0] = 0.
        return degs

    def deg2mm(self, deg):
        circumference_mm = self.ball_diam_mm * np.pi
        mm = deg * (circumference_mm / 360)
        return mm

    def is_circ(self, label):
        circlabels = [
            'heading',
            'side',
            'forw',
            'xstim',
            'servopos',
            'wstim',
        ]
        is_circ = np.array([label == circlabel for circlabel in circlabels]).any()
        return is_circ

    def get_matlab_signals(self):
        self.mat_filename = glob.glob(self.folder + os.path.sep + '*.mat')[0]
        mat = loadmat(self.mat_filename, squeeze_me=True)
        mat_time = mat['n'][0]
        temp = mat['n'][1]
        start_time = self.start_time.timestamp()
        mat_time = mat_time - start_time
        f = interp1d(mat_time, temp, kind='linear', bounds_error=False, fill_value=np.nan)
        self.df['upsampled_temp'] = f(self.df['t'])

    def get_boxcar_average(self, label, window_s):
        signal = self.df[label].values
        inds = np.arange(len(signal))
        half_window_s = window_s / 2.
        # bc an integer number of indices is used, the window is not necessarily exact
        half_window_ind = int(np.round(half_window_s * self.subsampling_rate))
        start = inds - half_window_ind
        end = inds + half_window_ind + 1
        # prevents wrapping, but means that some windows are shorter
        start[start < 0] = 0
        end[end > len(signal)] = len(signal)
        signal_out = np.zeros(len(signal))
        signal_out[:] = np.nan
        for i in inds:
            t0 = start[i]
            t1 = end[i]
            signal_out[i] = np.mean(signal[t0:t1])
        new_label = label + '_boxcar_average_' + str(window_s) + '_s'
        self.df[new_label] = signal_out


class Camera(object):
    """
      Class for handling camera videos (acquired using ROS video saver)
      -abf_time: stores time of each frame in abf time
      method is currently using a combination of camera triggers and time_stamps

    """

    def __init__(self, rec, camera_name='front'):
        self.camera_name = camera_name
        self.csv_filename = glob.glob(rec.folder + os.path.sep + '*timestamps.csv')[0]
        self.avi_filename = glob.glob(rec.folder + os.path.sep + '*.avi')[0]
        self.abf_start_time = rec.abf.start_time

        video = cv2.VideoCapture(self.avi_filename)
        self.n_frames = int(video.get(7))
        video.release()

        time_stamps_df = pd.read_csv(self.csv_filename)
        time_stamps_df['timestamp_s'] = time_stamps_df[' timestamp'] / (10 ** 9)
        time_stamps_df['abf_time'] = time_stamps_df['timestamp_s'] - self.abf_start_time.timestamp()
        self.n_time_stamps = len(time_stamps_df)

        if self.n_time_stamps != self.n_frames:
            print("Number of time stamps doesn't match number of frames!")
            return

        trig_signal = rec.abf.df_orig['camtrig']
        trigs = get_trig(trig_signal)
        t_trigs = rec.abf.df_orig['t'][trigs]
        idx = np.where(t_trigs < time_stamps_df['abf_time'].iloc[0])[0][-1]
        self.abf_time = t_trigs[idx:self.n_time_stamps].values


class Patch(object):
    """
       Class for handling for processing patching data

       -Detects spikes
       -Downsamples spikes and vm to behaviour sampling rate

       :param rec
       :detection_method : 'butterworth' (only method implemented atm)
       :detection_params : dictionnary

   """

    def __init__(self, rec, detection_method='butterworth', detection_params=None,
                 rec_params=None):
        self.detection_method = detection_method
        vm = rec.abf.df_orig['patch_1'].values
        sampling_rate = rec.abf.sampling_rate

        # default params for spike detection
        if detection_params is None:
            if self.detection_method == 'butterworth':
                self.detection_params = {
                    'freq_range': [100, 3000],  # frequency range for butterworth
                    'threshold': 0.5,  # threshold for spikes (after applying filter)
                    'min_distance_s': 0.005  # min distance between spikes in seconds
                }
        else:
            # weird things were happening when not making copy...
            self.detection_params = detection_params.copy()

        # rec specific overiding of default params
        if rec_params is not None:
            if rec.rec_name in rec_params.keys():
                params = rec_params[rec.rec_name]
                for key in params.keys():
                    self.detection_params[key] = params[key]

        sp = self.get_spikes(vm, sampling_rate, self.detection_method,
                             self.detection_params)

        vm_sp_subtracted = self.get_vm_sp_subtracted(vm, sp, sampling_rate, window_s=0.02)
        # modifies dataframes
        rec.abf.df_orig['sp'] = sp
        rec.abf.df_orig['vm_sp_subtracted'] = vm_sp_subtracted
        rec.abf.df['sp'] = self.downsample_to_behaviour(rec, 'sp')
        rec.abf.df['vm'] = self.downsample_to_behaviour(rec, 'patch_1')
        rec.abf.df['vm_sp_subtracted'] = self.downsample_to_behaviour(rec, 'vm_sp_subtracted')
        rec.abf.df['sp_rate'] = rec.abf.df['sp'] * sampling_rate

    def get_spikes(self, vm, sampling_rate, method,
                   detection_params):

        # applies butterworth bandpass filter
        # then finds local maxima that are above a certain threshold using sc.find_peaks
        # only includes spikes that are within a minimum distance see sc.find_peaks for details
        if method == 'butterworth':
            out = fc.butterworth(vm,
                                 passside='bandpass',
                                 cutoff=np.array(detection_params['freq_range']),
                                 sampling_freq=sampling_rate)
            min_distance = int(sampling_rate * detection_params['min_distance_s'])
            threshold = detection_params['threshold']
            peaks = sc.signal.find_peaks(out, height=threshold, distance=min_distance)
            sp_inds = peaks[0]
            sp = np.zeros(len(out))
            sp[sp_inds] = 1
        else:
            print('spike method does not exist')
            sp = None

        return sp

    def get_vm_sp_subtracted(self, vm, sp, sampling_rate, window_s):
        half_window_s = window_s / 2.
        inds = np.where(sp == 1)[0]
        half_window_ind = int(half_window_s * sampling_rate)
        start = inds - half_window_ind
        end = inds + half_window_ind + 1
        vm = vm.copy()
        for t0, t1 in zip(start, end):
            vm[t0:t1] = np.nan
        vm_sp_subtracted = vm
        return vm_sp_subtracted

    def downsample_to_behaviour(self, rec, label):

        signal = rec.abf.df_orig[label].values
        # these are indices used for behaviour, start of camtrig
        subsampled_indices = rec.abf.get_subsampled_indices()

        # downsamples by taking average from start of one cam trig to start of next camtrig
        ind_period = int(np.round(rec.abf.sampling_rate / rec.abf.subsampling_rate))
        indices = np.insert(subsampled_indices, -1, subsampled_indices[-1] + ind_period)
        start = indices[:-1]
        end = indices[1:]
        downsampled = np.array([np.nanmean(signal[start:end]) for start, end in zip(start, end)])

        return downsampled


# ------------------ 2. Imaging data ----------------------- #

class Image(object):
    """
        Class for handling TIF files

        :param folder: path to folder containing registered tiff file (must be only file ending in  "reg.tif ")
                       and mask file

    """

    def __init__(self, folder, single_z=False):

        self.folder = folder
        self.rec_name = folder.split('/')[-1]

        # finds tiff filename
        self.tiff_filename = glob.glob(folder + os.path.sep + self.rec_name + '_reg.tif')[0]

        # opens tiff file
        self.tif = tifffile.imread(self.tiff_filename)

        # reshapes tif to standard shape
        self.tif = self.reshape_tif(self.tif, single_z)

        # Extract acquisition info.
        self.tlen = self.tif.shape[0]
        self.zlen = self.tif.shape[1]
        self.n_channels = self.tif.shape[2]
        self.ylen = self.tif.shape[3]
        self.xlen = self.tif.shape[4]

        # Creates empty dataframe that will be used to store deltaF/F
        self.df = pd.DataFrame()

        # these variables can be set by Rec object
        self.volume_period = None
        self.volume_rate = None
        self.frame_rate = None

        # stores roi types pertaining to image
        self.roi_types = []

    def get_ROI(self, roi_type, celltypes=None):
        if celltypes is None:
            celltypes = {}
        # gets an ROI type
        roi_label_dict = {Bridge: 'pb', EllipsoidBody: 'eb', PairedStructure: 'ps', FanShapedBody: 'fb',
                          ArbitraryRois: 'a'}
        roi_label = roi_label_dict[roi_type]
        self.roi_types.append(roi_label)
        setattr(self, roi_label, roi_type(self.folder, self.tif, self.tlen, self.zlen,
                                          self.n_channels, celltypes, self.volume_period))
        roi = getattr(self, roi_label)
        roi_df = roi.process_data()
        roi_df = roi_df.add_prefix(roi_label + '_')
        self.df = pd.concat([self.df, roi_df], axis=1)
        # deletes tif image, since not necessary to keep in memory
        del roi.tif

    def is_circ(self, label):
        circlabels = [
            'phase',
        ]
        is_circ = np.array([re.match('.*' + circlabel + '*.', label) for circlabel in circlabels]).any()

        return is_circ

    def reshape_tif(self, tif, single_z=False):
        # TZCYX

        # Reshape tif so that multi-z and single-z tifs have the same shape dimension
        if len(tif.shape) == 3:
            newshape = list(tif.shape)
            newshape.insert(1, 1)
            tif = tif.reshape(tuple(newshape))

        # Reshape tif so that multi-channel and single-channel tifs have the same shape dimension
        # assumes that you will never do single plane dual color imaging
        # (this is indistinguishable from single color two plane imaging without reading metadata)
        if len(tif.shape) == 4:
            newshape = list(tif.shape)
            # TODO fix so not hardcoded
            # use for single color, multi z
            if single_z is False:
                newshape.insert(2, 1)
            else:
                # use for multi color single z
                newshape.insert(1, 1)
            tif = tif.reshape(tuple(newshape))

        return tif


class ROI(object):
    """
    Parent class for ROIs. Each ROI class must  have a process data function
    that returns a pandas dataframe where each row corresponds to an imaging acquisition timepoint
    and each column corresponds to some variable e.g. roi_1_deltaF/F

    :param celltypes: dictionary mapping channels to celltypes
                      e.g. {'c1':'EPG','c2':'PEN'}

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        self.folder = folder
        self.tif = tif
        self.tlen = tlen
        self.zlen = zlen
        self.n_channels = n_channels
        self.celltypes = celltypes
        self.volume_period = volume_period

    def get_mean_pixel_values(self, greyscales, mask_image):
        """
            Gets mean pixel value of each roi for each timepoint/volume (raw F) and stores in Pandas data frame
            e.g. "c1_roi1"

            :param greyscales: list of greyscale values of rois in mask_image
            :param mask_image: mask image with size (z,y,x).Currently assumes that you are using the same mask for both channels.

            :returns mean_pixel_vals: Pandas DataFrame

        """

        mean_pixel_values = dict()

        for i_channel in range(self.n_channels):
            channel_label = "c" + str(i_channel + 1)
            for i_roi, gs_val in enumerate(greyscales):
                roi_label = "roi_" + str(i_roi + 1)
                mask = (mask_image == gs_val)
                signal = self.tif[:, :, i_channel, :, :]
                if mask.sum():
                    # gets pixel values belonging to individual roi
                    roi_pixel_val = signal * mask
                    #  for each timpepoint get mean of pixel value of mask
                    # (averaged across total number of pixels in mask across all z slices)
                    mean_roi_pixel_val = np.nansum(roi_pixel_val.reshape(self.tlen, signal[0].size),
                                                   axis=1) / mask.sum()
                else:
                    # if gs value is not present in the mask then roi will be filled with nans
                    mean_roi_pixel_val = np.zeros(self.tlen)
                    mean_roi_pixel_val[:] = np.nan

                mean_pixel_values[channel_label + "_" + roi_label] = mean_roi_pixel_val.astype(np.float32)

        mean_pixel_values_df = pd.DataFrame(mean_pixel_values)

        return mean_pixel_values_df

    def normalize_over_time(self, signal, method='dF/F0', f0_frac=.05):
        """
            Normalizes mean pxiel value of an roi over time

            :param signal: Pandas Series with individual roi mean pixel values of length t
            :param method: string can be 'dF/F0' or zscore
            :param f0_frac: fraction used for f0 (only for dF/F0 mode)

            :returns norm_signal: Pandas Series with normalized signal
        """
        if method == 'dF/F0':
            sorted_signal = signal.sort_values(ascending=True, ignore_index=True)
            f0 = sorted_signal[:int(len(signal) * f0_frac)].mean()
            norm_signal = (signal - f0) / f0

        elif method == 'zscore':
            norm_signal = (signal - signal.mean()) / signal.std()

        elif method == 'maxmin':
            norm_signal = (signal - signal.min()) / signal.max()

        else:
            print("Normalization method not recognized")
            norm_signal = None

        return norm_signal

    def get_fft_spectrum(self, data):
        """
            Gets FFT phase and power spectrum

            :returns phase spectrum, power spectrum, sampling frequencies (as periods)

        """

        # used for zero padding data
        n = 100
        data = data.copy()
        # detrend data
        data -= data.mean()
        axlen = data.shape[1] * n
        # get FFT
        fourier = np.fft.fft(data, axlen, 1)
        amplitude = np.abs(fourier) ** 2
        power_spectrum = amplitude / amplitude.max()
        # sample frequencies
        freq = np.fft.fftfreq(axlen, 1. / n) / n
        period = (1. / freq)
        # phase spectrum
        phase_spectrum = np.angle(fourier)

        return phase_spectrum, power_spectrum, period

    def get_fft_phase_power(self, data, periodicity):
        """
            Gets phase and power at specified periodicity

            :returns phase (degrees) and power

        """
        phase_spectrum, power_spectrum, period = self.get_fft_spectrum(data)
        # get phase and power at specfified periodicity
        idx = np.where(period <= periodicity)[0][0]
        phase = -np.rad2deg(phase_spectrum[:, idx])
        power = power_spectrum[:, idx]

        return phase, power

    def get_pva(self, data):
        """
           Gets phase and amplitude using population vector average method
           :returns theta (degrees) and amplitude

        """
        data = data.copy()
        step = (2 * np.pi) / np.shape(data)[1]
        angles = np.arange(0, 2 * np.pi, step) + step / 2.
        # subtracts min from each row to remove negative dF/F (used as weights)
        data = (data.T - np.min(data, axis=1)).T
        data = (data.T / np.sum(data, axis=1)).T
        x, y = fc.polar2cart(data, angles)
        r, theta = fc.cart2polar(x.sum(axis=1), y.sum(axis=1))
        theta *= 180 / np.pi
        return theta, r

    def open_mask(self, mask_filename):
        mask_image = tifffile.imread(mask_filename)
        if len(mask_image.shape) == 2:
            newshape = list(mask_image.shape)
            # old, keep
            # newshape.insert(2, 1)
            newshape.insert(0, 1)
            mask_image = mask_image.reshape(tuple(newshape))

        return mask_image


class Bridge(ROI):
    """
    Child class of ROI

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        ROI.__init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period)
        # finds mask filename
        self.mask_filename = glob.glob(self.folder + os.path.sep + '*_pbmask.tif')[0]
        self.mask_image = self.open_mask(self.mask_filename)
        # gs values for each 18 glomeruli from left to right
        self.greyscales = [13, 25, 38, 51, 64, 76, 89, 102, 115, 128, 140, 154, 166, 179, 192, 205, 217, 230]

    def process_data(self):
        """
        :returns Pandas DataFrame
        """

        # gets mean pixel values for each roi
        mean_pixel_values = self.get_mean_pixel_values(self.greyscales, self.mask_image)

        # sets NaN for missing glomeruli (necessary since same mask is used for both celltypes)
        for channel, celltype in self.celltypes.items():
            if celltype == 'EPG':
                mean_pixel_values[channel + '_roi_1'] = np.nan
                mean_pixel_values[channel + '_roi_18'] = np.nan
            elif (celltype in ['PEN', 'IBSPSP']):
                mean_pixel_values[channel + '_roi_9'] = np.nan
                mean_pixel_values[channel + '_roi_10'] = np.nan

        # gets normalized signals
        self.f0_frac = 0.05
        df_f0 = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'dF/F0', self.f0_frac), axis=0)
        df_f0 = df_f0.add_suffix('_dF/F')

        zscore = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'zscore'), axis=0)
        zscore = zscore.add_suffix('_z')
        mean_pixel_values = mean_pixel_values.add_suffix('_F')
        df = pd.concat([mean_pixel_values, df_f0, zscore], axis=1)

        # periodicity used for each celltype
        self.periodicity_dic = {'EPG': 8.5, 'PEN': 8, 'PEG': 8.5}

        # gets phase using df_f0 signal
        for channel, celltype in self.celltypes.items():
            signal = df.filter(regex=channel + '.*._dF/F')
            continuous_bridge = self.get_continuous_bridge(signal, celltype)
            phase, power = self.get_fft_phase_power(continuous_bridge, self.periodicity_dic[celltype])
            phase = pd.Series(phase, name=channel + '_phase', dtype=np.float32)
            dphase = fc.circgrad(phase) * self.volume_period
            dphase = pd.Series(dphase, name=channel + '_dphase', dtype=np.float32)
            power = pd.Series(power, name=channel + '_power', dtype=np.float32)

            eb = self.get_eb_proj(continuous_bridge, celltype)
            pva_phase, r = self.get_pva(eb)
            pva_phase = pd.Series(pva_phase, name=channel + '_pva_phase', dtype=np.float32)
            pva_r = pd.Series(r, name=channel + '_pva_amplitude', dtype=np.float32)

            mean = np.nanmean(signal, axis=1)
            mean = pd.Series(mean, name=channel + '_mean_dF/F')
            max_min = np.nanmax(signal, axis=1) - np.nanmin(signal, axis=1)
            max_min = pd.Series(max_min, name=channel + '_max_min')

            df = pd.concat([df, phase, dphase, power, pva_phase, pva_r, mean, max_min], axis=1)

        return df

    def get_eb_proj(self, continuous_bridge, celltype):
        eb = np.zeros(continuous_bridge.shape)
        if celltype == 'EPG':
            eb[:, ::2] = continuous_bridge.iloc[:, 8:]
            eb[:, 1::2] = continuous_bridge.iloc[:, :8]
        return eb

    def get_continuous_bridge(self, signal, celltype):

        continuous_bridge = signal.copy()

        if celltype == 'EPG':
            continuous_bridge = continuous_bridge.iloc[:, 1:17]
        elif (celltype in ['PEN', 'IBSPSP']):
            continuous_bridge.iloc[:, 8] = (continuous_bridge.iloc[:, 0] + continuous_bridge.iloc[:, 16]) / 2.
            continuous_bridge.iloc[:, 9] = (continuous_bridge.iloc[:, 1] + continuous_bridge.iloc[:, 17]) / 2.
        elif (celltype == 'PEG'):
            continuous_bridge = continuous_bridge
        else:
            print("celltype not recognized")

        return continuous_bridge

    def interpolate_glomeruli(self, data, celltype, kind='cubic', dinterp=.1):
        tlen, period_inds = data.shape
        if celltype == 'EPG':
            wrap = np.zeros_like(data)
            wrap[:, 1:17] = data[:, 1:17]
            # fills in missing glomeruli
            wrap[:, [0, 17]] = data[:, [8, 9]]
            x = np.arange(0, period_inds, 1) + 0.5
            f = interp1d(x, wrap, kind, axis=-1)
            x_interp = np.arange(1, 17, dinterp) + dinterp / 2.
            row_interp = f(x_interp)

        else:
            print('celltype not recognized')

        return x_interp, row_interp

    def cancel_phase(self, gc, phase, offset, dinterp):
        period = 8
        period_inds = int(period / dinterp)

        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)

        x = np.arange(np.shape(gc)[1])
        left_ind = (x < (len(x) / 2))
        right_ind = (x >= (len(x) / 2))

        for i in range(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.zeros(len(x))
            row[:] = np.nan
            row[left_ind] = np.roll(gc[i, left_ind], shift)
            row[right_ind] = np.roll(gc[i, right_ind], shift)
            gc_nophase[i] = row

        return gc_nophase

    def get_mean_phase_nulled_activity(self, data, phase, offset=0, celltype='EPG'):
        x_interp, row_interp = self.interpolate_glomeruli(self, data, celltype=celltype, dinterp=0.1)
        gc_nophase = self.cancel_phase(self, row_interp, phase, offset=offset, dinterp=0.1)
        mean = np.mean(gc_nophase, axis=0)
        # downsample
        f = interp1d(x_interp, mean, kind='cubic', axis=-1)
        # hardcoded for EPGs
        x = np.arange(1, 17) + 0.5
        mean = f(x)
        return mean


class EllipsoidBody(ROI):
    """
    Child class of ROI

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        ROI.__init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period)
        # finds mask filename
        self.mask_filename = glob.glob(self.folder + os.path.sep + '*_ebmask.tif')[0]
        self.mask_image = self.open_mask(self.mask_filename)

        # make this an input variable!
        self.ncols = 16

        # same code to calculate greyscales as used in GUI
        step = int(255 / self.ncols)
        greyscales = np.arange(1, 255, step)
        if len(greyscales) > self.ncols:
            greyscales = greyscales[- self.ncols:]
        self.greyscales = greyscales

    def process_data(self):
        """
        :returns Pandas DataFrame
        """
        mean_pixel_values = self.get_mean_pixel_values(self.greyscales, self.mask_image)
        self.f0_frac = 0.05
        df_f0 = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'dF/F0', self.f0_frac), axis=0)
        df_f0 = df_f0.add_suffix('_dF/F')

        zscore = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'zscore'), axis=0)
        zscore = zscore.add_suffix('_z')

        mean_pixel_values = mean_pixel_values.add_suffix('_F')

        df = pd.concat([mean_pixel_values, df_f0, zscore], axis=1)

        for channel, celltype in self.celltypes.items():
            signal = df.filter(regex=channel + '.*._dF/F')
            phase, amplitude = self.get_pva(signal)
            dphase = fc.circgrad(phase) * self.volume_period
            phase = pd.Series(phase, name=channel + '_phase', dtype=np.float32)
            amplitude = pd.Series(amplitude, name=channel + '_amplitude', dtype=np.float32)
            dphase = pd.Series(dphase, name=channel + '_dphase', dtype=np.float32)
            df = pd.concat([df, phase, amplitude, dphase], axis=1)

        return df


class FanShapedBody(ROI):
    """
    Child class of ROI

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        ROI.__init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period)
        # finds mask filename
        # self.mask_filename = glob.glob(self.folder + os.path.sep + '*pbmask.tif')[0]
        # self.mask_image = skimage.external.tifffile.imread(self.mask_filename)
        self.mask_filename = glob.glob(self.folder + os.path.sep + '*_fbmask.tif')[0]
        self.mask_image = self.open_mask(self.mask_filename)

        # make this an input variable!
        self.ncols = 16

        # same code to calculate greyscales as used in GUI
        step = int(255 / self.ncols)
        greyscales = np.arange(1, 255, step)
        if len(greyscales) > self.ncols:
            greyscales = greyscales[- self.ncols:]
        self.greyscales = greyscales

    # self.greyscales = [gs for gs in np.unique(self.mask_image) if gs not in [0, 255]]

    def process_data(self):
        """
        :returns Pandas DataFrame
        """
        mean_pixel_values = self.get_mean_pixel_values(self.greyscales, self.mask_image)
        self.f0_frac = 0.05
        df_f0 = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'dF/F0', self.f0_frac), axis=0)
        df_f0 = df_f0.add_suffix('_dF/F')

        zscore = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'zscore'), axis=0)
        zscore = zscore.add_suffix('_z')

        mean_pixel_values = mean_pixel_values.add_suffix('_F')

        df = pd.concat([mean_pixel_values, df_f0, zscore], axis=1)

        for channel, celltype in self.celltypes.items():
            signal = df.filter(regex=channel + '.*._dF/F')
            phase, amplitude = self.get_pva(signal)
            dphase = fc.circgrad(phase) * self.volume_period
            phase = pd.Series(phase, name=channel + '_phase', dtype=np.float32)
            amplitude = pd.Series(amplitude, name=channel + '_pva_amplitude', dtype=np.float32)
            dphase = pd.Series(dphase, name=channel + '_dphase', dtype=np.float32)

            mean = np.nanmean(signal, axis=1)
            mean = pd.Series(mean, name=channel + '_mean_dF/F')
            max_min = np.nanmax(signal, axis=1) - np.nanmin(signal, axis=1)
            max_min = pd.Series(max_min, name=channel + '_max_min')

            df = pd.concat([df, phase, amplitude, dphase, mean, max_min], axis=1)

        return df

    def interpolate_wedges(self, data, kind='cubic', dinterp=.1):

        tlen, period_inds = data.shape

        # To extrapolate, we add two "fake gomeruli/column" at the edges
        wrap = np.zeros((data.shape[0], data.shape[1] + 2))
        wrap[:, 1:period_inds + 1] = data[:, :period_inds]
        wrap[:, [0, period_inds + 1]] = data[:, [-1, 0]]

        x = np.arange(-1, period_inds + 1, 1) + 0.5
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(0, period_inds, dinterp) + dinterp / 2.
        row_interp = f(x_interp)
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, offset):
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        for i in range(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.roll(gc[i], shift)
            gc_nophase[i] = row

        return gc_nophase

    def get_mean_phase_nulled_activity(self, data, phase, offset=0):
        x_interp, row_interp = self.interpolate_wedges(self, data, dinterp=0.1)
        gc_nophase = self.cancel_phase(self, row_interp, phase, offset=offset)
        mean = np.mean(gc_nophase, axis=0)
        # downsample
        f = interp1d(x_interp, mean, kind='cubic', axis=-1)
        x = np.arange(0, data.shape[1]) + 0.5
        mean = f(x)
        df = pd.DataFrame({'vals': mean, 'col': np.arange(len(mean))})
        return df


class PairedStructure(ROI):
    """
    Child class of ROI

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        ROI.__init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period)
        # finds mask filename
        self.mask_filename = glob.glob(self.folder + os.path.sep + '*reg_mask.tif')[0]
        self.mask_image = self.open_mask(self.mask_filename)
        # left roi should have  gs 13 and right  gs 230 (according to old code)
        # with new GUI left and right ROIs have gs 1 and 128
        self.greyscales = [gs for gs in np.unique(self.mask_image) if gs not in [0, 255]]
        if len(self.greyscales) > 2:
            print('There should be only two gs levels for paired structure!')

    # self.greyscales =  [13, 230]

    def process_data(self):
        """
        :returns Pandas DataFrame
        """
        mean_pixel_values = self.get_mean_pixel_values(self.greyscales, self.mask_image)
        self.f0_frac = 0.05
        df_f0 = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'dF/F0', self.f0_frac), axis=0)
        df_f0 = df_f0.add_suffix('_dF/F')

        zscore = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'zscore'), axis=0)
        zscore = zscore.add_suffix('_z')

        mean_pixel_values = mean_pixel_values.add_suffix('_F')

        df = pd.concat([mean_pixel_values, df_f0, zscore], axis=1)

        # gets right minus left for dF/F and z score
        for channel, celltype in self.celltypes.items():
            rml_dF_F = df[channel + '_roi_2_dF/F'] - df[channel + '_roi_1_dF/F']
            rml_dF_F = pd.Series(rml_dF_F, name=channel + '_rml_dF/F', dtype=np.float32)

            rpl_dF_F = df[channel + '_roi_2_dF/F'] + df[channel + '_roi_1_dF/F']
            rpl_dF_F = pd.Series(rpl_dF_F, name=channel + '_rpl_dF/F', dtype=np.float32)

            rml_z = df[channel + '_roi_2_z'] - df[channel + '_roi_1_z']
            rml_z = pd.Series(rml_z, name=channel + '_rml_z', dtype=np.float32)

            rpl_z = df[channel + '_roi_2_z'] + df[channel + '_roi_1_z']
            rpl_z = pd.Series(rpl_z, name=channel + '_rpl_z', dtype=np.float32)

            df = pd.concat([df, rml_dF_F, rml_z, rpl_dF_F, rpl_z], axis=1)

        return df


class ArbitraryRois(ROI):
    """
        Child class of ROI

    """

    def __init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period):
        ROI.__init__(self, folder, tif, tlen, zlen, n_channels, celltypes, volume_period)
        # finds mask filename
        self.mask_filename = glob.glob(self.folder + os.path.sep + '*reg_mask.tif')[0]
        self.mask_image = self.open_mask(self.mask_filename)
        self.greyscales = [gs for gs in np.unique(self.mask_image) if gs not in [0, 255]]

    def process_data(self):
        """
        :returns Pandas DataFrame
        """
        mean_pixel_values = self.get_mean_pixel_values(self.greyscales, self.mask_image)
        self.f0_frac = 0.05
        df_f0 = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'dF/F0', self.f0_frac), axis=0)
        df_f0 = df_f0.add_suffix('_dF/F')

        zscore = mean_pixel_values.apply(lambda x: self.normalize_over_time(x, 'zscore'), axis=0)
        zscore = zscore.add_suffix('_z')

        mean_pixel_values = mean_pixel_values.add_suffix('_F')

        df = pd.concat([mean_pixel_values, df_f0, zscore], axis=1)

        return df


# ------------------ 3. Combining behaviour and imaging data ----------------------- #

class Rec(object):

    def __init__(self, folder, trim_times=None, bh_type=Walk, bh_kws=None, ext='.abf', light=True,
                 patch=False, patch_kws=None, camera=False, hdf5=False, delete_columns=None):

        self.folder = folder
        self.rec_name = folder.split('/')[-1]
        # TODO still not right place for tim_times
        self.trim_time = None
        if trim_times is not None:
            if self.rec_name in trim_times.keys():
                self.trim_time = trim_times[self.rec_name]

        # finds file that end with extension of behaviour file
        abf_filename = glob.glob(self.folder + os.path.sep + '*' + ext)[0]

        # process abf
        if bh_kws is None:
            bh_kws = {}
        self.abf = bh_type(folder, abf_filename, **bh_kws)

        if patch:
            self.patch = Patch(self, **patch_kws)

        if camera:
            if len(glob.glob(self.folder + os.path.sep + '*.avi')) > 0:
                self.camera = Camera(self)

        # this needs to go after Patch is created if you want to do boxcar average on sp_rate
        if self.abf.boxcar_average is not None:
            for label, window_s in self.abf.boxcar_average.items():
                self.abf.get_boxcar_average(label, window_s)

        # TODO put this in a better place, placed here because removing NaNs before downsampling ephys doenst work
        ## removes first few rows that contain NaNs (due to shift) and reset index
        ## used to be in Walk Class. We dont want to drop all NaNs bc vm_sp_subtracted has NaNs
        self.abf.df = self.abf.df.dropna(subset=['x', 'y'])
        self.abf.df.reset_index(drop=True, inplace=True)

        if self.trim_time is not None:
            self.trim(self.trim_time)

        if hdf5:
            self.make_hdf5()

        if delete_columns is not None:
            self.abf.df.drop(columns=delete_columns, inplace=True)

        if light == True:
            del self.abf.df_orig
        # if light is a list, it will only keep those columns
        elif light:
            self.abf.df_orig = self.abf.df_orig.filter(items=light)

    def trim(self, trim_time):
        # might be slow?!
        new_df = self.abf.df_orig.query(f" t>= {trim_time[0]}& t<={trim_time[1]}").copy()
        self.abf.df_orig = new_df
        self.abf.df_orig.reset_index(drop=True, inplace=True)

        new_df = self.abf.df.query(f" t>= {trim_time[0]}& t<={trim_time[1]}").copy()
        self.abf.df = new_df
        self.abf.df.reset_index(drop=True, inplace=True)

    def make_hdf5(self):
        # saves all downsampled channels and original sampling of patching channel
        # as HDF5
        hdf5_name = self.folder + os.sep + self.rec_name + '.h5'
        for col in self.abf.df.columns:
            self.abf.df[col].to_hdf(hdf5_name, key=col, mode='a')
        self.abf.df_orig['t'].to_hdf(hdf5_name, key='t_orig', mode='a')
        self.abf.df_orig['patch_1'].to_hdf(hdf5_name, 'patch_1', mode='a')
        self.abf.df_orig['camtrig'].to_hdf(hdf5_name, 'camtrig', mode='a')
        if hasattr(self, 'camera'):
            pd.Series(self.camera.abf_time).to_hdf(hdf5_name, 'camera_time', mode='a')


class ImagingRec(Rec):
    """
        Class for handling a single imaging recording: ABF and Image object

        :param folder: path of folder containing both ABF file and registered TIFF file
                       (must be only file to end in "reg.tif")
        :param bh_type: ABF  child class
        :param ext: extension for abf file (.abf or .edr)
        :param bh_kws: dictionary of keyword arguments passed to ABF  child class (optional)
        :param roi_types: list of ROI child class e.g. [Bridge,EllipsoidBody]
        :param roi_kws: dictionary of keyword arguments passed to ROI  child class
        :param photostim: boolean,if True stores frame mean of x-galvo, y-gavo position and pockels
        :param photostim_kws:  dictionary of keyword arguments passed to PhotoStim class
        :param concat_df: boolean, if True it will concatenate downsampled behaviour df with imaging df
        :param light: boolean, if true it will delete tiff image
                      and non downsanmled abf data frame from Rec object
    """

    def __init__(self, folder, trim_times=None, bh_type=Walk, bh_kws=None, ext='.abf', roi_kws=None, roi_types=None,
                 photostim=False, photostim_kws=None, concat_df=True, light=True, delete_columns=None, single_z=False):

        Rec.__init__(self, folder, trim_times=trim_times, bh_type=bh_type, bh_kws=bh_kws, ext=ext, light=False,
                     delete_columns=delete_columns)

        # read TIF
        self.im = Image(self.folder, single_z)

        # renames columns if necessary
        clock_labels = fc.contains(['clock', '2pframest'], self.abf.df_orig.columns)
        self.abf.df_orig = self.abf.df_orig.rename(columns={clock_labels: "clock"})

        # gets indices of start and end of each imaging frame (this is the Y-galvo flyback on the Scientifica)
        im_frame_start = get_trig(self.abf.df_orig['clock'], 4, falling=False)
        im_frame_end = get_trig(self.abf.df_orig['clock'], 4, falling=True)

        # removes frames where pockels were not on, this is necessary on Prairie system
        # (and maybe Scientifica sometimes?) since there is a dropped frame during piezo flyback
        # im_frame_start_pockels_on, im_frame_end_pockels_on = self.get_pockels_on(im_frame_start, im_frame_end,self.abf.df_orig['pockels'])

        # removed this functionality since won't work with photostimulation
        # (where mean pockel during frame can be very low), but now will fail if there are frames dropped due to
        # piezo flyback
        im_frame_start_pockels_on = im_frame_start
        im_frame_end_pockels_on = im_frame_end

        # gets timepoint of start and end of each imaging frame
        im_frame_start_pockels_on_t = self.abf.df_orig['t'][im_frame_start_pockels_on]
        im_frame_end_pockels_on_t = self.abf.df_orig['t'][im_frame_end_pockels_on]

        # gets start and end of each volume (uses number of z slices from tiff shape)
        im_vol_start_t = im_frame_start_pockels_on_t[::self.im.zlen]
        im_vol_end_t = im_frame_end_pockels_on_t[self.im.zlen - 1::self.im.zlen]

        # set t=0 when 2P acquisition begins
        t_acq_start = im_vol_start_t.iloc[0]
        self.abf.df['t'] -= t_acq_start
        im_vol_start_t -= t_acq_start
        im_vol_end_t -= t_acq_start

        im_vol_start_t.reset_index(drop=True, inplace=True)
        im_vol_end_t.reset_index(drop=True, inplace=True)

        # stores index of start and end of each volume acquisition (used for downsampling)
        self.im.df['vol_start_ind'] = [np.where(self.abf.df['t'].values >= start)[0][0] for start in im_vol_start_t]
        self.im.df['vol_end_ind'] = [np.where(self.abf.df['t'].values <= end)[0][-1] for end in im_vol_end_t]
        # t is the middle of volume acquisition
        self.im.df['t'] = (im_vol_start_t + im_vol_end_t) / 2.

        # stores average volume and frame rate
        self.im.volume_period = np.mean(np.diff(self.im.df['t']))
        self.im.volume_rate = 1. / self.im.volume_period
        self.im.frame_rate = self.im.volume_rate * self.im.zlen

        # stores average x and y galvo position for each frame of each volume
        if photostim:
            frame_mean_x_galvo_pos = [np.mean(self.abf.df_orig['x_galvo'].values[start:end]) for start, end in
                                      zip(im_frame_start_pockels_on, im_frame_end_pockels_on)]
            frame_mean_y_galvo_pos = [np.mean(self.abf.df_orig['y_galvo'].values[start:end]) for start, end in
                                      zip(im_frame_start_pockels_on, im_frame_end_pockels_on)]
            frame_mean_pockels = [np.mean(self.abf.df_orig['pockels'].values[start:end]) for start, end in
                                  zip(im_frame_start_pockels_on, im_frame_end_pockels_on)]

            frame_mean_x_galvo_pos = np.array([frame_mean_x_galvo_pos[i::self.im.zlen] for i in range(self.im.zlen)])
            frame_mean_y_galvo_pos = np.array([frame_mean_y_galvo_pos[i::self.im.zlen] for i in range(self.im.zlen)])
            frame_mean_pockels = np.array([frame_mean_pockels[i::self.im.zlen] for i in range(self.im.zlen)])
            for z in range(self.im.zlen):
                self.im.df['mean_x_galvo_pos_slice_' + str(z)] = frame_mean_x_galvo_pos[z, :]
                self.im.df['mean_y_galvo_pos_slice_' + str(z)] = frame_mean_y_galvo_pos[z, :]
                self.im.df['mean_pockels_slice_' + str(z)] = frame_mean_pockels[z, :]

        # process tiff for each ROI type
        if roi_types is None:
            roi_types = [Bridge]
        if roi_kws is None:
            roi_kws = {}
        for roi_type in roi_types:
            self.im.get_ROI(roi_type, **roi_kws)

        if photostim_kws is None:
            photostim_kws = {}
        if photostim:
            self.photostim = PhotoStim(self, **photostim_kws)

        if concat_df:
            downsampled_df = self.get_downsampled_df()
            downsampled_df = downsampled_df.rename(columns={'t': 't_abf'})
            self.df = pd.concat([downsampled_df, self.im.df], axis=1)

        if light:
            del self.abf.df_orig
            del self.im.tif

    def get_pockels_on(self, frame_start, frame_end, pockels):
        """
            checks whether pockel signal is above some threshold during a frame acquisition

        """
        iterable = ((np.mean(pockels.values[start:end]) > 0.01) for start, end in zip(frame_start, frame_end))
        is_on = np.fromiter(iterable, bool)
        frame_start_pockels_on = frame_start[is_on]
        frame_end_pockels_on = frame_end[is_on]

        return frame_start_pockels_on, frame_end_pockels_on

    def downsample_to_imaging(self, label, lag_ms=0, metric=None, **kwargs):
        """
            This function will downsample a single abf channel (e.g. dforw) to imaging volume rate
            with lag_ms

        """
        if metric is None:
            if self.abf.is_circ(label):
                metric = fc.circmean
            else:
                metric = np.nanmean

        ind_lag = int(np.round(lag_ms / 1000. * self.abf.subsampling_rate))

        iterable = (metric(self.abf.df[label].values[start - ind_lag:end - ind_lag], **kwargs) for start, end in
                    zip(self.im.df['vol_start_ind'], self.im.df['vol_end_ind']))
        downsampled = np.fromiter(iterable, dtype=np.float32)

        return downsampled

    def get_downsampled_df(self):
        """
            This function downsamples all abf channels to imaging volume rate and creates new df
            Note: downsampled "t" will be different from im.df['t],even with lag_ms = 0. This because the number
            of behaviour frames in a volume might not be evenly spaced (whereas im.df['t] is middle of volume acquisition)


        """
        downsampled_df = pd.DataFrame()

        for column_name in list(self.abf.df.columns):
            downsampled_df[column_name] = self.downsample_to_imaging(column_name, lag_ms=0)

        return downsampled_df


class PhotoStim(object):
    """
    Class for handling photostimulation experiments that are based on moving the location of scanfields
    in the multiple ROI mode of ScanImage

    :param rec: Rec object
    :param galvo_label: name of im.df column to use for clustering (e.g. "mean_x_galvo_pos_slice_1"),
    -> TO DO: infer based on MATLAB output file
    :param photostim_scanfield_z: slice plane of photostim scanfield
    -> TO DO: infer based on MATLAB output file

    Adds columns to rec.im.df:
        -scanfield_config_name: name of scanfield configuration e.g. "left", "right", "control"
        -"_photostim": percentage of pixel overlap between structure roi and photostimulation ROI
        -trial_id:
    e.g. added column names might "ps_roi_1_photostim"

    Requires a .mat file with same name as rec_name which stores scanfield information

    """

    def __init__(self, rec, galvo_label=None, photostim_scanfield_z=1, scanfield_configs=None):
        self.rec = rec
        self.photostim_scanfield_z = photostim_scanfield_z
        self.galvo_label = galvo_label
        self.photo_stim_masks = {}
        self.scanfields = self.get_scanfields()
        if scanfield_configs is None:
            self.scanfield_configs = list(self.scanfields['photostimScanfield'].keys())
        else:
            self.scanfield_configs = scanfield_configs
        self.n_scanfield_configs = len(self.scanfield_configs)
        # adds scanfield_configuration to im.df
        self.add_scanfield_config()
        # for each roi of each roi_type, stores fraction of pixels inside stimulation roi as a new column in im.df
        self.rec.im.df = self.rec.im.df.groupby(['scanfield_config_name']).apply(self.set_pct_roi)

    def get_scanfields(self):
        """
        :returns scanfields: a dictionary of dictionaries
        the most nested dictionary stores the properties of a single scanfield:
            -pixelToRefTransform
            -pixelResolutionXY
            -centerXY
            -z (not yet implemented)
        e.g. :
        scanfields['photostimScanfield']['left']['centerXY'] -> array([-1.062 , -0.3802])
        scanfields['imScanfield']['pixelResolutionXY'] -> array([128,  64])

        """
        self.mat_filename = self.rec.folder + os.path.sep + self.rec.rec_name + '.mat'
        mat = loadmat(self.mat_filename, squeeze_me=True)
        self.trials = mat['trials']
        # TODO remove
        # dealing with silly mistake
        self.trials = np.array([''.join(trial.split('_')) for trial in self.trials], dtype='object')
        mat.pop('trials')
        keys = [i for i in list(mat.keys()) if '__' not in i]

        scanfield_names = [key.split('_')[0] for key in keys]
        scanfield_names = np.unique(scanfield_names)

        scanfields = {'imScanfield': {},
                      'photostimScanfield': {}}

        for scanfield_name in scanfield_names:
            if scanfield_name != 'imScanfield':
                scanfields['photostimScanfield'][scanfield_name] = {}

        for key in keys:
            scanfield_name = key.split('_')[0]
            scanfield_property_name = key.split('_')[-1]
            if scanfield_name == 'imScanfield':
                scanfields['imScanfield'][scanfield_property_name] = mat[key]
            else:
                scanfields['photostimScanfield'][scanfield_name][scanfield_property_name] = mat[key]

        return scanfields

    def add_scanfield_config(self):
        # clusters mean_x_galvo of specific slice
        # n_clusters should be number of different scanfield configurations in recording
        vals = self.rec.im.df[self.galvo_label].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_scanfield_configs)
        kmeans.fit(vals)
        clusters = kmeans.predict(vals)
        # maps clusters to of to scanfield configurations number
        _, idx = np.unique(clusters, return_index=True)
        unique_clusters = clusters[np.sort(idx)]
        self.rec.im.df['scanfield_cluster'] = clusters

        # maps cluster  to trial name (e.g. 'control','left','right)
        # wasnt able to map galvo position voltage to scanfield_config easily
        # uses order of trials saved by MATLAB object
        _, idx = np.unique(self.trials, return_index=True)
        unique_trials = self.trials[np.sort(idx)]
        unique_trials = np.insert(unique_trials, 0, 'control')
        clusters_to_trials = dict(zip(unique_clusters, unique_trials))
        self.rec.im.df['scanfield_config_name'] = self.rec.im.df['scanfield_cluster'].map(clusters_to_trials)

        # might want to move this outside of this file since usually this sort of stuff is
        # computed in analysis_plot
        self.rec.im.df['photostim_trial_id'] = fc.number_islands(self.rec.im.df['scanfield_config_name'] != 'control')

    def set_pct_roi(self, data):
        scanfield_config = data['scanfield_config_name'].values[0]
        # for scanfield_config in self.scanfield_configs:
        for roi_type in self.rec.im.roi_types:
            pct_in_roi = self.get_pct_in_roi(roi_type, scanfield_config)
            for iroi, pct in enumerate(pct_in_roi):
                data[roi_type + '_roi_' + str(iroi + 1) + '_photostim'] = pct

        return data

    def get_pct_in_roi(self, roi_type, scanfield_config):
        # returns fraction of pixels inside the stimulation roi
        # for each roi of an roi type (e.g. each column of fb)
        image_scanfield = self.scanfields['imScanfield']
        image_size = image_scanfield['pixelResolutionXY']
        image_transform = image_scanfield['pixelToRefTransform']
        image_pixel_coords = self.get_ref_coords(image_size, image_transform)

        photostim_scanfield = self.scanfields['photostimScanfield'][scanfield_config]
        photo_stim_size = photostim_scanfield['pixelResolutionXY']
        # # TODO remove, silly mistake, is approx right
        # if 'pixelToRefTransform' not in photostim_scanfield:
        #     photostim_scanfield['pixelToRefTransform']=np.array([[ 0.0108 ,  0.0063     , 0.2085],
        #                                                          [ -0.0630    ,  0.0108 , -0.1487],
        #                                                          [ 0.     ,  0.     ,  1.     ]])
        #

        photo_stim_transform = photostim_scanfield['pixelToRefTransform']
        photostim_pixel_coords = self.get_ref_coords(photo_stim_size, photo_stim_transform)

        def get_corners(coords):
            corners = np.array([
                coords[0, 0],
                coords[-1, 0],
                coords[-1, -1],
                coords[0, -1],
            ])
            return corners

        photostim_pixel_coords = np.reshape(photostim_pixel_coords, [photo_stim_size[0], photo_stim_size[1], 2])
        photostim_corners = get_corners(photostim_pixel_coords)
        photostim_polygon = Polygon(photostim_corners)

        image_pixel_coords = np.reshape(image_pixel_coords, [image_size[0], image_size[1], 2])

        is_photostim = np.zeros([np.shape(image_pixel_coords)[0], np.shape(image_pixel_coords)[1]]).astype(bool)

        for index in np.ndindex(image_size[0], image_size[1]):
            pixel = image_pixel_coords[index]
            p = Point(pixel)
            is_photostim[index] = p.within(photostim_polygon)

        # gets ROI volume mask of imaging scanfield
        mask = getattr(self.rec.im, roi_type).mask_image[:, :image_size[1], :image_size[0]]

        # gets volume mask of photostim scanfield
        photo_stim_mask = np.zeros(np.shape(mask)).astype(bool)
        photo_stim_mask[self.photostim_scanfield_z] = np.reshape(is_photostim, image_size).T

        self.photo_stim_masks[scanfield_config] = photo_stim_mask

        pct_in_roi = []
        for gs in getattr(self.rec.im, roi_type).greyscales:
            roi = mask == gs
            is_photostim_in_roi = roi * photo_stim_mask
            pct = np.sum(is_photostim_in_roi) / np.sum(roi)
            pct_in_roi.append(pct)
        return pct_in_roi

    def get_ref_coords(self, dim, transform):
        # does affine tranformation

        # dim is size of scanfield (n rows, n columns)
        # transform is pixelToRefTransform (3,3 matrix) or some other transformation matrix
        # returns N,2 array containing pixel coordinates in reference coordinate system (row angle, col angle)

        coords = []
        # starts at 1 because of MATLAB indexing
        for x in np.arange(1, dim[0] + 1):
            for y in np.arange(1, dim[1] + 1):
                pixel_coords = np.array([x, y, 1])
                ref_coords = transform @ pixel_coords
                #         new_coords = np.linalg.inv(imageT) @ ref_coords
                #         coords.append(list(new_coords))
                coords.append(list(ref_coords))
        coords = np.array(coords)
        return (coords[:, :2])


# ------------------ 4. Handling multiple recordings ----------------------- #

class Experiment(object):
    """
       Class for handling multiple Rec objects

       :param rec_names: list of tuples containing rec names.
                         Each tuple should contain the recordings names of a single fly
                         e.g.
                         rec_names=[
                            # fly 1
                            ('2020_02_02_0001',),  # rec 1
                            # fly 2
                            ('2020_02_03_0001', # rec 1
                            '2020_02_03_0003') # rec 2
                        ]

                        Note: if a fly has a single recording you need the comma to make it a tuple
        :param parent_folder: path to parent folder
        :param merge_df: boolean, if True, it will merge
        :param genotype: if not not None, adds genotype column to merged_abf_df and merged_im_df
    """

    def __init__(self, rec_names, rec_type=ImagingRec,
                 parent_folder=None, merge_df=True, genotype=None, subfolders=True, **kwargs):

        self.params = kwargs
        self.genotype = genotype
        self.rec_type = rec_type

        if parent_folder is not None:
            self.parent_folder = parent_folder
        else:
            self.parent_folder = os.getcwd() + '/'
        # Nested dictionary storing Rec objects, first layer is fly e.g. "fly_1"
        # Second layer is recording e.g. "2020_03_25_0001"
        self.fly_data_dict = {}
        for ifly, fly in enumerate(rec_names):
            self.fly_data_dict['fly_' + str(ifly + 1)] = {}
            for rec in fly:
                date_folder = '_'.join(rec.split('_')[:-1])
                if subfolders:
                    folder = self.parent_folder + date_folder + '/' + rec
                else:
                    folder = self.parent_folder + date_folder
                self.fly_data_dict['fly_' + str(ifly + 1)].update({rec: self.rec_type(folder, **kwargs)})

        if merge_df:
            self.merge_df()

        if self.genotype is not None:
            self.add_genotype()

    def __iter__(self):
        """
            Iterates over the Experiment object will yield a namedtuple tuple for each Rec object
            (with fly_id, rec_name and rec object)

        """

        RecInfo = namedtuple('RecInfo', 'fly_id rec_name rec')

        return iter([RecInfo(ifly, irec, rec) for ifly, fly in sorted(self.fly_data_dict.items()) for irec, rec in
                     sorted(fly.items())])

    def __getitem__(self, item):
        """
           Gets rec by using its name Experiment[rec_name]

        """
        return [rec for ifly, fly in sorted(self.fly_data_dict.items()) for rec_name, rec in sorted(fly.items()) if
                rec_name == item][0]

    def add_rec(self, ifly, rec, **kwargs):
        date_folder = '_'.join(rec.split('_')[:-1])
        folder = self.parent_folder + date_folder + '/' + rec
        if ifly not in self.fly_data_dict:
            self.fly_data_dict[ifly] = {rec: self.rec_type(folder, **kwargs)}
        else:
            self.fly_data_dict[ifly].update({rec: self.rec_type(folder, **kwargs)})

    def merge_df(self):
        """
            Creates  merged pandas DataFrame of all recordings :
            one for behaviour and one for imaging (if rec_type is ImagingRec)

        """

        self.merged_abf_df = pd.DataFrame()
        if self.rec_type == ImagingRec:
            self.merged_im_df = pd.DataFrame()

        for rec in self:

            abf_df = rec.rec.abf.df.copy()
            fly_id = int(rec.fly_id.split('_')[-1])
            abf_df.insert(0, 'fly_id', fly_id)
            abf_df.insert(1, 'rec_name', rec.rec_name)
            self.merged_abf_df = self.merged_abf_df.append(abf_df, ignore_index=True)

            if self.rec_type == ImagingRec:
                # If behaviour has been downsampled and concatenated with im_df, the concatenated df will be used
                if hasattr(rec.rec, 'df'):
                    im_df = rec.rec.df.copy()
                else:
                    im_df = rec.rec.im.df.copy()
                im_df.insert(0, 'fly_id', fly_id)
                im_df.insert(1, 'rec_name', rec.rec_name)
                self.merged_im_df = self.merged_im_df.append(im_df, ignore_index=True)

    def downsample_merged(self, labels, lag_ms, metric=None, suffix=None, **kwargs):

        """
            Downsamples a signal from merged_abf_df and adds it to merged_im_df

            :param labels: list of column names in merged_abf_df to be downsampled
            :param metric: metric used for downsampling e.g. nan_mean
            :param lag_ms: lag is milliseconds
            :param suffix: suffix to append to new label
            :**kwargs: keyword arguments passed to metric function

        """

        def downsample(data, self, labels, metric, lag_ms, **kwargs):
            rec_name = data['rec_name'].iloc[0]
            ind_lag = int(np.round(lag_ms / 1000. * self[rec_name].abf.subsampling_rate))
            abf_df = self.merged_abf_df.query("rec_name=='" + rec_name + "'")

            for label in labels:
                if metric is None:
                    if self[rec_name].abf.is_circ(label):
                        metric = fc.circmean
                    else:
                        metric = np.nanmean

                downsampled = [metric(abf_df[label].values[start - ind_lag:end - ind_lag], **kwargs) for start, end in
                               zip(data['vol_start_ind'], data['vol_end_ind'])]

                if suffix:
                    label = label + '_' + suffix

                data[label] = downsampled

            return data

        self.merged_im_df = self.merged_im_df.groupby('rec_name').apply(downsample, self, labels, metric, lag_ms)

    def add_genotype(self):
        self.merged_abf_df.insert(0, 'genotype', self.genotype)
        if self.rec_type == ImagingRec:
            self.merged_im_df.insert(0, 'genotype', self.genotype)


def quickload_experiment(pickle_fname, rec_names, reprocess=False, exp_kws=None, **kwargs):
    """
        Loads a pickeled Experiment object. For a given fly, if rec_names contains recs that were not pickeled, it will add them to the
        Experiment object. If Experiment objects contains recs that are not in rec_names, it will remove them.
        If the fly order is changed, data will be re-processed

        :param pickle_fname: pickle will be made and saved as pickle_fname (needs to end in ".pkl")
        :param rec_names: list of tuples containing rec names. Each tuple should contain the recordings names of a single fly
                          if None, it will load an existing pickle (equivalent to just using joblib.load(pickle_fname))
        :param parent_folder: parent_folder for Experiment
        :param reprocess: dtype=bool, if True re-process all data

        :returns experiment: Experiment object


    """

    # might not need to have all these as default
    if exp_kws is None:
        exp_kws = {
            'rec_type': None,
            'parent_folder': None,
            'merge_df': True,
            'genotype': None}

    # if there is no pickle with the pickle_fname in current directory or if reprocess is True,
    # then process data and make new pickle

    if (not os.path.isfile(pickle_fname)) | reprocess:
        experiment = Experiment(rec_names, **exp_kws, **kwargs)
        joblib.dump(experiment, pickle_fname)

    else:
        # loads pickeled Experiment object
        experiment = joblib.load(pickle_fname)

        if rec_names is not None:
            # loops through each fly
            for ifly, fly in enumerate(rec_names):
                fly_id = 'fly_' + str(ifly + 1)
                # gets fly's rec_names that were already processed
                old_recs_names = list(experiment.fly_data_dict.get(fly_id, {}).keys())
                # gets fly's rec_names that user wants to keep/process
                input_recs_names = list(rec_names[ifly])
                # get rec_names to be removed
                remove_recnames = list(set(old_recs_names) - set(input_recs_names))
                # get rec_names to be added
                add_recnames = list(set(input_recs_names) - set(old_recs_names))
                # add/removes recs
                for rec_name in remove_recnames:
                    experiment.fly_data_dict[fly_id].pop(rec_name)
                for rec_name in add_recnames:
                    experiment.add_rec(fly_id, rec_name, **kwargs)

            # makes new merged df
            experiment.merge_df()
            if experiment.genotype is not None:
                experiment.add_genotype()
            # saves new Experiment object as pickle
            joblib.dump(experiment, pickle_fname)

    return experiment


# ------------------ Useful functions ----------------------- #

def get_trig(signal, trigger=1, falling=False):
    """
        Returns indices of rising or falling triggers
    """
    if falling:
        trigs = np.where(np.diff((signal < trigger).astype(int)) > 0)[0] - 1
    else:
        trigs = np.where(np.diff((signal > trigger).astype(int)) > 0)[0] + 1
    return trigs
