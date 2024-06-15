__author__ = 'Raymond Chia'

import warnings
import os
from os.path import join, exists
import time
from pathlib import Path
import mne

from math import ceil
import numpy as np

from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.ndimage import uniform_filter1d

from bsl import StreamPlayer, StreamRecorder, datasets
from bsl import StreamReceiver
from bsl.lsl import resolve_streams
from bsl.utils import Timer
from bsl.triggers import MockTrigger, TriggerDef

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import patches

from utils import load_model

DEBUG = True
# STREAM_NAME = 'EEGLAB'
STREAM_NAME = 'X.on-102802-0085 (Allen-Dell23)'
WEIGHTS_FNAME = None
N_CLASSES = 4

CLASS_LBLS = ['Low', 'Med', 'High', 'VHigh']
CLASS_DICT = {k: v for k, v in zip(range(N_CLASSES), CLASS_LBLS)}

DEMO_TIME = 10 # seconds

def eeg_rest_stream():
    fif_file = datasets.eeg_resting_state.data_path()

    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)
    ''' below only necessary if repeat != 1 '''
    time.sleep(10)
    player.stop()
    print(player)

def eeg_rest_trigger():
    tdef = TriggerDef()
    tdef.add("rest", 1)

    player = StreamPlayer(stream_name, fif_file, trigger_def=tdef)
    player.start()
    print(player)


    time.sleep(5)
    player.stop()
    print(player)

def eeg_mock_trigger(tdef):
    trigger = MockTrigger()
    trigger.signal(1)

    player = StreamPlayer(stream_name, fif_file, trigger_def=tdef)
    player.start()
    print(player)


    time.sleep(5)
    player.stop()
    print(player)

def recorder():
    fif_file = datasets.eeg_resting_state.data_path()
    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)

    streams = [stream.name for stream in resolve_streams()]
    print(streams)
    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    recorder = StreamRecorder(record_dir, fname='example-rest-state')
    recorder.start()
    print(recorder)

    trigger = MockTrigger()
    trigger.signal(1)

    time.sleep(2)
    recorder.stop()
    print(recorder)

    player.stop()
    print(player)

    fname = join(record_dir, 'fif',
                 'example-resting-state-StreamPlayer-raw.fif')
    time.sleep(3)
    if exists(fname):
        raw = mne.io.read_raw_fif(fname, preload=True)
        print(raw)
        events = mne.find_events(raw, stim_channel='TRIGGER')
        print(events)

def stream_receiver():
    fif_file = datasets.eeg_resting_state.data_path()
    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)

    streams = [stream.name for stream in resolve_streams()]
    print(streams)
    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    recorder = StreamRecorder(record_dir, fname='example-rest-state')
    recorder.start()
    print(recorder)

    trigger = MockTrigger()
    trigger.signal(1)

    time.sleep(2)
    recorder.stop()
    print(recorder)

    player.stop()
    print(player)

    fname = join(record_dir, 'fif',
                 'example-resting-state-StreamPlayer-raw.fif')
    time.sleep(3)
    if exists(fname):
        raw = mne.io.read_raw_fif(fname, preload=True)
        print(raw)
        events = mne.find_events(raw, stim_channel='TRIGGER')
        print(events)

class SignalAnimator():
    '''
    Plots signals specified by channel_names and channels_to_plot
    '''
    def __init__(self, fig, channels_to_plot=[0, 1, 2], buffer_samples=500,
                 channel_names=['Fpz', 'Cz', 'P1']):
        self.channels_to_plot = channels_to_plot
        self.channel_names = channel_names
        self.n_channels = len(channels_to_plot)
        self.buffer_samples = buffer_samples

        self.fig = fig
        self.cmap = plt.get_cmap('tab20')

        self.raw_cmap_nums = np.arange(0, 2*self.n_channels, step=2)
        self.dsp_cmap_nums = np.arange(1, 2*self.n_channels+1, step=2)

        ncols = 2

        self.raw_axs, self.dsp_axs = [], []
        for i, (ch, ch_name) in enumerate(zip(channels_to_plot, channel_names)):
            self.raw_axs.append(plt.subplot2grid(shape=(self.n_channels, ncols*2),
                                                 loc=(ch, 0), colspan=1))
            self.dsp_axs.append(plt.subplot2grid(shape=(self.n_channels, ncols*2),
                                                 loc=(ch, 1), colspan=1))

            self.raw_axs[i].set_ylabel(ch_name)

        self.raw_lines = self.create_lines(self.raw_axs, cmap_nums=self.raw_cmap_nums)
        self.dsp_lines = self.create_lines(self.dsp_axs, cmap_nums=self.dsp_cmap_nums)

        self.raw_axs[0].set_title("Raw Signal")
        self.dsp_axs[0].set_title("Processed Signal")

        for raw_ax, dsp_ax in zip(self.raw_axs[:-1], self.dsp_axs[:-1]):
            raw_ax.set_xticklabels([])
            dsp_ax.set_xticklabels([])

        self.raw_axs[-1].set_xlabel("Time (s)")
        self.dsp_axs[-1].set_xlabel("Time (s)")

        self.raw_y_lim = np.zeros(self.n_channels)
        self.dsp_y_lim = np.zeros(self.n_channels)

    def create_lines(self, axs, cmap_nums=[0, 1, 2]):
        lines = []
        for i, ax in enumerate(axs):
            lines.append(
                ax.plot(
                    np.empty(self.buffer_samples),
                    np.empty(self.buffer_samples),
                    c=self.cmap(cmap_nums[i])
                )
            )
        return lines

    def animate(self, i, xdata, raw_data, dsp_data):
        for j, (raw_line, dsp_line) in enumerate(zip(self.raw_lines,
                                                     self.dsp_lines)):
            ch = self.channels_to_plot[j]
            raw_out = raw_data[:, ch]
            dsp_out = dsp_data[:, ch]

            raw_line[0].set_data(xdata, raw_out)
            dsp_line[0].set_data(xdata, dsp_out)

            self.raw_axs[j].set_xlim([xdata[0], xdata[-1]])
            self.dsp_axs[j].set_xlim([xdata[0], xdata[-1]])

            if self.raw_y_lim[j] < np.abs(np.max(raw_out)):
                self.raw_y_lim[j] = np.abs(np.max(raw_out))
            if self.dsp_y_lim[j] < np.abs(np.max(dsp_out)):
                self.dsp_y_lim[j] = np.abs(np.max(dsp_out))

            self.raw_axs[j].set_ylim([-self.raw_y_lim[j], self.raw_y_lim[j]])
            self.dsp_axs[j].set_ylim([-self.dsp_y_lim[j], self.dsp_y_lim[j]])

class ClassAnimator():
    '''
    Plots softmax classification probability to bar plot
    '''
    def __init__(self, fig, n_classes=N_CLASSES, class_lbls=CLASS_LBLS):
        self.n_classes = n_classes
        self.class_lbls = class_lbls
        self.fig = fig

        self.cls_ax = plt.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1)

        self.cls_ax.set_xlabel("Class Label")
        self.cls_ax.set_ylabel("Probabilities")
        self.cls_ax.set_ylim([0, 1])
        self.cls_ax.set_xlim([-0.5, self.n_classes-0.5])
        self.cls_ax.set_xticks(np.arange(self.n_classes))
        self.cls_ax.set_xticklabels(self.class_lbls)

        self.barplot = self.cls_ax.bar(np.arange(self.n_classes),
                                       np.zeros(self.n_classes))

    def animate(self, i, probs):
        assert len(probs)==len(self.barplot), "Incorrect number of "\
                f"probability inputs. Requires size ({len(self.barplot)},)"
        for bar, prob in zip(self.barplot, probs):
            bar.set_height(prob)

def create_bandpass_filter(low, high, fs, n):
    """
    Create a bandpass filter using a butter filter of order n.

    Parameters
    ----------
    low : float
        The lower pass-band edge.
    high : float
        The upper pass-band edge.
    fs : float
        Sampling rate of the data.
    n : int
        Order of the filter.

    Returns
    -------
    sos : array
        Second-order sections representation of the IIR filter.
    zi_coeff : array
        Initial condition for sosfilt for step response steady-state.
    """
    # Divide by the Nyquist frequency
    bp_low = low / (0.5 * fs)
    bp_high = high / (0.5 * fs)
    # Compute SOS output (second order sections)
    sos = butter(n, [bp_low, bp_high], btype="band", output="sos")
    # Construct initial conditions for sosfilt for step response steady-state.
    zi_coeff = sosfilt_zi(sos).reshape((sos.shape[0], 2, 1))

    return sos, zi_coeff

class Buffer():
    """
    A buffer containing filter data and its associated timestamps.

    Parameters
    ----------
    buffer_duration : float
        Length of the buffer in seconds.
    sr : bsl.StreamReceiver
        StreamReceiver connected to the desired data stream.
    """

    def __init__(self, buffer_duration, sr, model, n_classes=N_CLASSES,
                 model_seq_len=200):
        # Store the StreamReceiver in a class attribute
        self.sr = sr

        # Retrieve sampling rate and number of channels
        self.fs = int(self.sr.streams[stream_name].sample_rate)
        self.nb_channels = len(self.sr.streams[stream_name].ch_list) - 1
        self.n_classes = n_classes
        self.model = model

        self.model_seq_len = model_seq_len
        buffer_duration_samples = ceil(buffer_duration * self.fs)

        if self.model_seq_len > buffer_duration_samples:
            self.model_seq_len = int(buffer_duration//1.5)
            warnings.warn("Sequence length too large, amending to"\
                          f"{self.model_seq_len}")

        # Define duration
        self.buffer_duration = buffer_duration
        self.buffer_duration_samples = buffer_duration_samples

        # Create data array
        self.timestamps = np.zeros(self.buffer_duration_samples)
        self.data = np.zeros((self.buffer_duration_samples,
                              self.nb_channels))
        # For demo purposes, let's store also the raw data
        self.raw_data = np.zeros((self.buffer_duration_samples,
                                  self.nb_channels))

        # classifier output
        self.class_data = np.zeros((self.buffer_duration_samples, n_classes))

        # Create filter BP (1, 15) Hz and filter variables
        self.sos, self.zi_coeff = create_bandpass_filter(5.0, 10.0, self.fs, n=2)
        self.zi = None

    def update(self):
        """
        Update the buffer with new samples from the StreamReceiver. This method
        should be called regularly, with a period at least smaller than the
        StreamReceiver buffer length.
        """
        # Acquire new data points
        self.sr.acquire()
        data_acquired, ts_list = self.sr.get_buffer()
        self.sr.reset_buffer()
        print("buff len list: ", len(ts_list))

        if len(ts_list) == 0:
            return  # break early, no new samples

        # Remove trigger channel
        data_acquired = data_acquired[:, 1:]

        # Filter acquired data
        if self.zi is None:
            # Initialize the initial conditions for the cascaded filter delays.
            self.zi = self.zi_coeff * np.mean(data_acquired, axis=0)
        data_filtered, self.zi = sosfilt(self.sos, data_acquired, axis=0, zi=self.zi)

        # Roll buffer, remove samples exiting and add new samples
        self.timestamps = np.roll(self.timestamps, -len(ts_list))
        self.timestamps[-len(ts_list) :] = ts_list

        print("timestamp fs: ", np.mean(1/np.diff(self.timestamps)))
        self.data = np.roll(self.data, -len(ts_list), axis=0)
        self.data[-len(ts_list) :, :] = data_filtered
        self.raw_data = np.roll(self.raw_data, -len(ts_list), axis=0)
        self.raw_data[-len(ts_list) :, :] = data_acquired

        data_to_model = np.expand_dims(self.data[-self.model_seq_len:],
                                       axis=0)
        preds = self.model.predict(data_to_model, verbose=0)
        self.class_data = np.roll(self.class_data, -len(preds), axis=0)
        self.class_data[-len(preds) :, :] = preds

def tutorial_model(weights_fname:str=None, n_classes=N_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='selu', name='InputLayer'),
        tf.keras.layers.Dense(1000, activation='relu', name='Hidden1'),
        tf.keras.layers.Dense(100, activation='relu', name='Hidden2'),
        tf.keras.layers.Dense(10, activation='relu', name='Hidden3'),
        tf.keras.layers.Reshape((-1,), name='Reshape'),
        tf.keras.layers.Dense(n_classes, activation='softmax',
                              name='OutputLayer'),
    ])
    if weights_fname is not None:
        assert exists(weights_fname), "Cannot find the weights file, is this"\
                " the right path?"
        print("Loading weights... ", end='')
        model.load_weights(weights_fname)
        print("Success!")

    return model

def get_receiver(bufsize=1, winsize=0.5, stream_name="StreamPlayer"):
    receiver = StreamReceiver(bufsize=bufsize, winsize=winsize,
                              stream_name=stream_name)
    time.sleep(winsize)  # wait to fill LSL inlet.
    return receiver

def real_time_stream(stream_name, model_fname=None):
    demo_time = DEMO_TIME # seconds
    n_classes = N_CLASSES

    buffer_duration = 5
    model_seq_len = 200 # seconds, sequence length for model
    learning_rate = 1e-3
    loss = 'mse'
    pred_buff_display = 15
    buff_refresh = 0.1

    bufsize = 1
    winsize = 0.5

    if model_fname is not None:
        model = load_model(model_fname)
    else:
        model = tutorial_model(WEIGHTS_FNAME, n_classes=n_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        # After the model is created, we then config the model with losses and metrics
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[loss])

    # define the receiver
    # receiver = get_receiver(bufsize=1, winsize=0.5, stream_name=stream_name)
    receiver = get_receiver(bufsize=bufsize, winsize=winsize,
                            stream_name=stream_name)

    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    trigger = MockTrigger()
    trigger.signal(1)

    buffer = Buffer(buffer_duration, receiver, model,
                    model_seq_len=model_seq_len)

    timer = Timer()
    while timer.sec() <= buffer_duration:
        buffer.update()
    timer.reset()

    # fig, ax = plt.subplots(2,2, figsize=(12,7), sharex=True)
    fig = plt.figure(figsize=(12,7))

    sig_anim = SignalAnimator(fig)
    cls_anim = ClassAnimator(fig)

    idx_last_plot = 1
    dt, t0 = 0, 0
    # set interactive
    while timer.sec() <= demo_time:
        t0 = timer.sec()
        buffer.update()
        if dt >= buff_refresh:
            sig_anim.animate(idx_last_plot, buffer.timestamps, buffer.raw_data,
                             buffer.data)
            cls_anim.animate(idx_last_plot, buffer.class_data[-1])

            idx_last_plot+=1
            plt.pause(0.01)

            # refresh done, reset timer
            dt = 0

        # increment timer
        n_dt = timer.sec() - t0
        dt += n_dt

    del receiver
    # recorder.stop()
    plt.tight_layout()
    plt.show()
    model.summary()

if __name__ == '__main__':
    #######################
    # TESTING
    #######################
    # start the streamer for testing
    if DEBUG:
        stream_name = "StreamPlayer"
        fif_file = datasets.eeg_resting_state.data_path()
        player = StreamPlayer(stream_name, fif_file)
        player.start()
        print(player)
    else:
        stream_name = STREAM_NAME

    streams = [stream.name for stream in resolve_streams()]
    print("List of available streams\n: ", streams)

    assert stream_name in streams, f"Cannot find {stream_name}"

    # run the real-time classifier
    if DEBUG:
        stream_name = 'StreamPlayer'

    print("Streaming from: ", stream_name)
    real_time_stream(stream_name)

    if DEBUG:
        player.stop()
        print(player)
