from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import librosa
from config import *


def Evaluate_model(y_true, y_pred):
    # site: https://machinelearningcoban.com/2017/08/31/evaluation/
    labels = np.array(y_true)
    print('test accuracy = ', accuracy_score(labels, y_pred), ' %')

    print(classification_report(labels, y_pred))

    cnf_matrix = confusion_matrix(labels, y_pred)
    print('Confusion matrix:\n', cnf_matrix)
    print('\nAccuracy:', np.diagonal(cnf_matrix).sum() / cnf_matrix.sum())


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


def main(PATH=None, testflag: bool = False, clf=None, jso: dict or None = None) -> dict or None:
    if PATH is None:  # run test
        testflag = True
        PATH = librosa.util.find_files(Test_path.data_path)
    if debug:
        PATH = PATH[:3]
    dict_test: dict = {}
    labels = []
    for p in PATH:
        file = p.split(os.sep)[-1]
        labels.append(p.split(os.sep)[-1].split('_')[0])  # contains filenames (when method is calles from gradio) or labels (when running test)
        features = feature_extract(p)
        dict_test[file] = features
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
    assert clf is not None
    test_Y_hat = clf.predict_proba(data)
    best_4 = np.argsort(test_Y_hat, axis=1)[:,-mult_res:]

    result_s = list(best_4)
    assert len(result_s) == len(PATH) == len(labels)

    confirm_each_result_is_sorted(test_Y_hat, result_s)

    if testflag:  # run test
        Evaluate_model(labels, [class_names[int(result_s[i][0])] for i in range(len(labels))])
    else:  # method is calles from gradio
        for i, key in enumerate(labels):
            if key not in jso: jso[key] = list()
            result = result_s[i]
            for top4 in result:
                ypre = class_names[int(top4)]
                if ypre not in jso[key]: jso[key].append(ypre)
    return jso


if __name__ == '__main__':
    PATH = librosa.util.find_files('/home/zaibachkhoa/Documents/Music-Genre-Classification-From-Audio-Files/Music_Instrument_Classification/dataset/valid/')
    jso = dict()
    for clf in model_listobj:
        main(PATH=PATH, testflag=True, clf=clf, jso=jso)
