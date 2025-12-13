import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import scipy
import time
import collections
import itertools
import librosa
import pickle

from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

bas_pa = '/media/zaibachkhoa/New Volume/__h2jsc_NESTECH/NSynth'
train_dir = f'{bas_pa}/nsynth-train/audio/'  # directory to training data and json file
valid_dir = f'{bas_pa}/nsynth-valid/audio/'  # directory to training data and json file
test_dir = f'{bas_pa}/nsynth-test/audio/'  # directory to training data and json file


def study_train_data():
    # đếm số lượng mẫu của từng loại nhạc cụ
    df_train_raw = pd.read_json(path_or_buf='nsynth-train/examples.json', orient='index')
    n_class_train = df_train_raw['instrument_family'].value_counts(ascending=True)

    # lấy ngẫu nhiên 5000 mẫu mỗi loại
    df_train_sample = df_train_raw.groupby('instrument_family', as_index=False,  # group by instrument family
                                           group_keys=False).apply(lambda df: df.sample(5000))  # number of samples
    # drop the synth_lead from the training dataset
    df_train_sample = df_train_sample[df_train_sample['instrument_family'] != 9]

    df_train_sample['instrument_family'].value_counts().reindex(np.arange(0, len(n_class_train), 1)).plot(kind='bar')
    plt.title("Instrument Family Distribution of Sampled Data: Training")
    plt.xlabel('Instrument Family')
    plt.ylabel('Number of Samples in Dataset')

    filenames_train = df_train_sample.index.tolist()
    with open('DataWrangling/filenames_train.pickle', 'wb') as f:
        pickle.dump(filenames_train, f)

    df_valid = pd.read_json(path_or_buf='nsynth-valid/examples.json', orient='index')
    # save the valid file index as list
    filenames_valid = df_valid.index.tolist()
    # save the list to a pickle file
    with open('DataWrangling/filenames_valid.pickle', 'wb') as f:
        pickle.dump(filenames_valid, f)

    # extract the filenames from the testing dataset
    df_test = pd.read_json(path_or_buf='nsynth-test/examples.json', orient='index')
    # save the test file index as list
    filenames_test = df_test.index.tolist()
    # save the list to a pickle file
    with open('DataWrangling/filenames_test.pickle', 'wb') as f:
        pickle.dump(filenames_test, f)


def feature_extract(file):
    """
    Define function that takes in a file an returns features in an array
    """

    # get wave representation
    y, sr = librosa.load(file)

    # determine if instruemnt is harmonic or percussive by comparing means
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic) > np.mean(y_percussive):
        harmonic = 1
    else:
        harmonic = 0

    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # temporal averaging
    mfcc = np.mean(mfcc, axis=1)

    # get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # temporally average spectrogram
    spectrogram = np.mean(spectrogram, axis=1)

    # compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    # temporally average chroma
    chroma = np.mean(chroma, axis=1)

    # compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis=1)

    return [harmonic, mfcc, spectrogram, chroma, contrast]


def instrument_code(filename):
    """
    Function that takes in a filename and returns instrument based on naming convention
    """
    class_names = ['bass', 'brass', 'flute', 'guitar',
                   'keyboard', 'mallet', 'organ', 'reed',
                   'string', 'synth_lead', 'vocal']

    for name in class_names:
        if name in filename:
            return class_names.index(name)
    else:
        return None


def extract_future_of_testdata():
    # create dictionary to store all test features
    dict_test = {}
    # loop over every file in the list
    for file in filenames_test:
        # extract the features
        features = feature_extract(test_dir + file + '.wav')  # specify directory and .wav
        # add dictionary entry
        dict_test[file] = features

    # convert dict to dataframe
    features_test = pd.DataFrame.from_dict(dict_test, orient='index',
                                           columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

    # extract mfccs
    mfcc_test = pd.DataFrame(features_test.mfcc.values.tolist(), index=features_test.index)
    mfcc_test = mfcc_test.add_prefix('mfcc_')

    # extract spectro
    spectro_test = pd.DataFrame(features_test.spectro.values.tolist(), index=features_test.index)
    spectro_test = spectro_test.add_prefix('spectro_')

    # extract chroma
    chroma_test = pd.DataFrame(features_test.chroma.values.tolist(), index=features_test.index)
    chroma_test = chroma_test.add_prefix('chroma_')

    # extract contrast
    contrast_test = pd.DataFrame(features_test.contrast.values.tolist(), index=features_test.index)
    contrast_test = chroma_test.add_prefix('contrast_')

    # drop the old columns
    features_test = features_test.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

    # concatenate
    df_features_test = pd.concat([features_test, mfcc_test, spectro_test, chroma_test, contrast_test],
                                 axis=1, join='inner')

    targets_test = []
    for name in df_features_test.index.tolist():
        targets_test.append(instrument_code(name))

    df_features_test['targets'] = targets_test

    # save the dataframe to a pickle file
    with open('DataWrangling/df_features_test.pickle', 'wb') as f:
        pickle.dump(df_features_test, f)


def extract_future_of_traindata():
    start_train = time.time()

    # create dictionary to store all test features
    dict_train = {}
    # loop over every file in the list
    for file in filenames_train:
        # extract the features
        features = feature_extract(train_dir + file + '.wav')  # specify directory and .wav
        # add dictionary entry
        dict_train[file] = features

    end_train = time.time()
    print('Time to extract {} files is {} seconds'.format(len(filenames_train), end_train - start_train))

    features_train = pd.DataFrame.from_dict(dict_train, orient='index',
                                            columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

    # extract mfccs
    mfcc_train = pd.DataFrame(features_train.mfcc.values.tolist(),
                              index=features_train.index)
    mfcc_train = mfcc_train.add_prefix('mfcc_')

    # extract spectro
    spectro_train = pd.DataFrame(features_train.spectro.values.tolist(),
                                 index=features_train.index)
    spectro_train = spectro_train.add_prefix('spectro_')

    # extract chroma
    chroma_train = pd.DataFrame(features_train.chroma.values.tolist(),
                                index=features_train.index)
    chroma_train = chroma_train.add_prefix('chroma_')

    # extract contrast
    contrast_train = pd.DataFrame(features_train.contrast.values.tolist(),
                                  index=features_train.index)
    contrast_train = chroma_train.add_prefix('contrast_')

    # drop the old columns
    features_train = features_train.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

    # concatenate
    df_features_train = pd.concat([features_train, mfcc_train, spectro_train, chroma_train, contrast_train],
                                  axis=1, join='inner')
    df_features_train.head()

    targets_train = []
    for name in df_features_train.index.tolist():
        targets_train.append(instrument_code(name))

    df_features_train['targets'] = targets_train

    with open('DataWrangling/df_features_train.pickle', 'wb') as f:
        pickle.dump(df_features_train, f)


def extract_future_of_valdata():
    start_valid = time.time()

    # create dictionary to store all test features
    dict_valid = {}
    # loop over every file in the list
    for file in filenames_valid:
        # extract the features
        features = feature_extract(valid_dir + file + '.wav')  # specify directory and .wav
        # add dictionary entry
        dict_valid[file] = features

    end_valid = time.time()
    print('Time to extract {} files is {} seconds'.format(len(filenames_valid), end_valid - start_valid))

    features_valid = pd.DataFrame.from_dict(dict_valid, orient='index',
                                            columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

    # extract mfccs
    mfcc_valid = pd.DataFrame(features_valid.mfcc.values.tolist(),
                              index=features_valid.index)
    mfcc_valid = mfcc_valid.add_prefix('mfcc_')

    # extract spectro
    spectro_valid = pd.DataFrame(features_valid.spectro.values.tolist(),
                                 index=features_valid.index)
    spectro_valid = spectro_valid.add_prefix('spectro_')

    # extract chroma
    chroma_valid = pd.DataFrame(features_valid.chroma.values.tolist(),
                                index=features_valid.index)
    chroma_valid = chroma_valid.add_prefix('chroma_')

    # extract contrast
    contrast_valid = pd.DataFrame(features_valid.contrast.values.tolist(),
                                  index=features_valid.index)
    contrast_valid = chroma_valid.add_prefix('contrast_')

    # drop the old columns
    features_valid = features_valid.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

    # concatenate
    df_features_valid = pd.concat([features_valid, mfcc_valid, spectro_valid, chroma_valid, contrast_valid],
                                  axis=1, join='inner')

    targets_valid = []
    for name in df_features_valid.index.tolist():
        targets_valid.append(instrument_code(name))

    df_features_valid['targets'] = targets_valid

    with open('DataWrangling/df_features_valid.pickle', 'wb') as f:
        pickle.dump(df_features_valid, f)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

