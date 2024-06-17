__author__ = 'Raymond Chia'
'''
Provides several functions for ease of use including sliding window process, 
model persistence, and model evaluation among others.
'''
import re
from os.path import join
from os import makedirs
import numpy as np
import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score,
    RocCurveDisplay, confusion_matrix
)
from sklearn.metrics import auc as auc_metric
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from pyxdf import load_xdf

from models import *
from configs import N_CLASSES, CKPT_DIR

def get_xdf_data(fname):
    (data_dict, marker_dict), header = load_xdf(fname)
    return (data_dict, marker_dict), header

def get_xdf_fs(data_dict):
    fs = data_dict['info']['effective_srate']
    return fs

def get_channel_names(data_dict):
    ch_df = pd.DataFrame(data_dict['info']['desc'][0]['channels'][0]['channel'])
    ch_df = ch_df.explode(ch_df.columns.values.tolist())

    ch_names = ch_df['label'].to_dict()
    return ch_names

def get_marker_freq_data(marker_dict, re_str="\d+\.\d+"):
    '''
    Returns the SSVEP marker labels (in frequency) and time stamps for each
    event.

    :param marker_dict: dictionary input from xdf file
    :type marker_dict: dict
    :param re_str: regex string
    :type re_str: str

    :return: the SSVEP marker event labels and timestamps
    :return type: tuple
    '''
    marker_timestamps = marker_dict['time_stamps']
    marker_labels = marker_dict['time_series']

    marker_labels = np.array([float(re.findall(re_str, val[0]).pop()) for val in
                              marker_labels])

    marker_labels = np.append(marker_labels, np.nan)
    marker_timestamps = np.append(marker_timestamps, marker_timestamps[-1]+5)
    return marker_labels, marker_timestamps

def get_marker_data(marker_dict, re_str="open"):
    '''
    Returns the marker labels (ssvep or eyes closed) and time stamps for each
    event.

    :param marker_dict: dictionary input from xdf file
    :type marker_dict: dict

    :return: the marker event labels and timestamps
    :return type: tuple
    '''
    marker_timestamps = marker_dict['time_stamps']
    marker_labels = marker_dict['time_series']

    tmp = []
    time_tmp = []
    for val, t in zip(marker_labels, marker_timestamps):
        if 'look' not in val[0]: continue

        if re_str in val[0]:
            tmp.append(0)
        else:
            tmp.append(1)

        time_tmp.append(t)

    marker_labels = np.array(tmp)
    marker_timestamps = np.array(time_tmp)

    marker_labels = np.append(marker_labels, np.nan)
    marker_timestamps = np.append(marker_timestamps, marker_timestamps[-1]+5)
    return marker_labels, marker_timestamps

def get_eeg_data(data_dict):
    data_timestamps = data_dict['time_stamps']
    data_array = data_dict['time_series']
    return data_array, data_timestamps

def get_windows(vec, window_size, window_shift):
    ''' Vectorised approach to sliding window
    https://towardsdatascience.com/
    fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    array: vec to slide
    window_size: window length
    window_shift: step stride len

    Create a rightmost vector as [0, V, 2V, ...].
    '''
    max_time = len(vec)-1
    sub_windows = (
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(max_time, step=window_shift), 0).T
    )
    i = [i for i, win in enumerate(sub_windows) if max_time in win][0]

    try:
        return vec[sub_windows[:i+1]]
    except IndexError:
        return vec[sub_windows[:i]]

def split_dataset(x, y, test_size=0.2, random_state=42, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, **kwargs)
    return x_train, x_test, y_train, y_test

def do_one_hot(y, max_categories=N_CLASSES, **kwargs):
    ohe = OneHotEncoder(max_categories=max_categories, **kwargs)
    ohe.fit(y)
    return ohe.transform(y), ohe

def save_model(model, model_fname, prefix=None):
    '''
    Saves model weights and parameters for persistence. If saving tensorflow
    model, ensure only the model is passed as an argument, not the class.

    :param model: model from the models.py file
    :type model: tf or scikit-learn model
    :param model_fname: model filename
    :type model_fname: str
    :param prefix: a prefix for model filename
    :type prefix: str
    '''
    makedirs(CKPT_DIR, exist_ok=True)
    if prefix is None:
        fname = join(CKPT_DIR, model_fname)
    else:
        fname = join(CKPT_DIR, '_'.join((prefix, model_fname)))

    if isinstance(model, tf.keras.Model):
        suffix = '.h5'
        fname += suffix
        model.save(fname)
    else:
        suffix = '.joblib'
        fname += suffix
        with open(fname, 'wb') as f:
            joblib.dump(model, f)

    print("saved to ", fname)

def load_model(model_fname):
    '''
    Loads model weights and parameters from previous saves. Ensure the full
    filename is specified.

    :param model_fname: model filename
    :type model_fname: str

    :return: loaded model
    :return type: tf or scikit-learn model
    '''
    if 'h5' in model_fname:
        model = tf.keras.saving.load_model(model_fname)
    elif 'joblib' in model_fname:
        with open(model_fname, 'rb') as f:
            model = joblib.load(f)

    return model

def plot_roc_curve(lbls, preds):
    '''
    Plots the ROC Curve. Useful for multiclass classficiation performance
    evaluation. Ideal classifier shows an ROC Curve that rejects all false
    positives and accepts only true positives.

    :param lbls: ground truth labels
    :type lbls: numpy.ndarray or numpy like array
    :param preds: predicted probabilities from model output, not the integer
    class predictions
    :type preds: numpy.ndarray or numpy like array
    '''
    def get_data(lbls, preds, average=None):
        if average != 'micro':
            # If not weighted averaging, then independently calculate false and
            # true positive rates
            if lbls.shape[0] > lbls.shape[1]:
                lbls = lbls.T

            if preds.shape[0] > preds.shape[1]:
                preds = preds.T

            fpr, tpr, auc = [], [], []
            for lbl, pred in zip(lbls, preds):
                fpr_tmp, tpr_tmp, _ = roc_curve(lbl, pred)
                fpr.append(fpr_tmp)
                tpr.append(tpr_tmp)
                auc.append(roc_auc_score(lbl, pred))
            if average == 'macro':
                fpr_grid = np.linspace(0, 1, 300)
                mu_tpr = np.zeros_like(fpr_grid)
                for f, t in zip(fpr, tpr):
                    mu_tpr += np.interp(fpr_grid, f, t)

                mu_tpr /= lbls.shape[0]
                fpr = fpr_grid
                tpr = mu_tpr
                auc = auc_metric(fpr, tpr)
        else:
            fpr, tpr, _ = roc_curve(lbls.ravel(), preds.ravel())
            auc = roc_auc_score(lbls, preds, average=average)
        return fpr, tpr, auc

    # Get outputs from averaging methods
    mi_fp, mi_tp, mi_auc = get_data(lbls, preds, average='micro')
    ma_fp, ma_tp, ma_auc = get_data(lbls, preds, average='macro')

    # Get outputs for each class
    fp_list, tp_list, auc_list = get_data(lbls, preds)

    # Plot
    plt.plot(mi_fp, mi_tp, label=f'micro-avg (auc = {mi_auc})')
    plt.plot(ma_fp, ma_tp, label=f'macro-avg (auc = {ma_auc})')
    for i, (fp, tp, auc) in enumerate(zip(fp_list, tp_list, auc_list)):
       plt.plot(fp, tp, label=f'class {i} (auc = {auc})', linestyle='dotted')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance level (auc = 0.5)')

    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

def show_classifier_performance(lbls, preds):
    print(classification_report(lbls, preds))

def plot_confusion_matrix(lbls, preds, labels=['0', '1', '2', '3', '4']):
    cm = confusion_matrix(lbls, preds)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.show(block=False)

def plot_channel_transform(time, freq, mag):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pcolormesh(time, freq, mag, cmap='viridis', vmin=0)
    ax.set_title("Transform Plot")
    return fig, ax
