import numpy as np
import pandas as pd
import librosa, pickle
from config import *


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
    for name in class_names:
        if name in filename:
            return class_names.index(name)
    else:
        return None


def main(PATH=None, clf=None, jso: dict or None = None) -> dict:
    dict_test: dict = {}
    for p in PATH:
        file = p.split(os.sep)[-1]
        features = feature_extract(p)
        dict_test[file] = features
    labels = dict_test.keys()   # contains filenames
    features_test = pd.DataFrame.from_dict(
        dict_test,
        orient='index',
        columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast']
    )

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
    data = pd.concat(
        [features_test, mfcc_test, spectro_test, chroma_test, contrast_test],
        axis=1,
        join='inner'
    )
    test_Y_hat = clf.predict(data)
    result = list(test_Y_hat)
    assert jso is not None
    assert len(result) == len(PATH) == len(labels)
    for i, key in enumerate(labels):
        res: int = int(result[i])
        ypre = class_names[res]
        if key not in jso: jso[key] = list()
        if ypre not in jso[key]: jso[key].append(ypre)
    return jso


if __name__ == '__main__':
    main()
