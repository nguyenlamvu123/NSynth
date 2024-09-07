from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

from datetime import datetime, timedelta


def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1

'''
Last Acceptable Values

min_silence_length = 0.3
silence_threshold = 1e-3
step_duration = 0.03/10

'''


def main(
        input_file,
        min_silence_length=0.6,  # The minimum length of silence at which a split may occur [seconds]. Defaults to 3 seconds.
        silence_threshold=1e-4,  # The energy level (between 0.0 and 1.0) below which the signal is regarded as silent.
        step_duration=0.03/10,  # The amount of time to step forward in the input file after calculating energy. Smaller value = slower, but more accurate silence detection. Larger value = faster, but might miss some split opportunities. Defaults to (min-silence-length / 10.).
) -> str:
    input_filename = input_file
    window_duration = min_silence_length
    if step_duration is None:
        step_duration = window_duration / 10.
    else:
        step_duration = step_duration

    sample_rate, samples = wavfile.read(filename=input_filename, mmap=True)

    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )

    window_energy = (energy(w) / max_energy for w in tqdm(
        signal_windows,
        total=int(len(samples) / float(step_size))
    ))

    window_silence = (e > silence_threshold for e in window_energy)
    window_silence_list = tuple(window_silence)
    num_tr = window_silence_list.count(True); print(num_tr, 'True values')  # giá trị num_tr (số lượng True) thấp hơn nhiều lần num_fa, lí tưởng là về 0 đối với nhạc không lời
    num_fa = window_silence_list.count(False); print(num_fa, 'False values')  # giá trị num_fa (số lượng False) thấp hơn nhiều lần num_tr, lí tưởng là về 0 đối với nhạc có lời
    return '' if num_tr < 1 / 2 * len(window_silence_list) else 'vocal'
    """
nhạc không lời:
    ################# nonvocal3.wav
    66285 True values
    62849 False values
    ################# nonvocal1.wav
    0 True values
    62475 False values
    ################# nonvocal2.wav
    0 True values
    100133 False values
    ################# bass_electronic_027-026-025.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-028-025.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-028-127.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-028-050.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-025-100.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-026-100.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-025-075.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-027-025.wav
    0 True values
    1136 False values
    ################# bass_electronic_027-025-025.wav
    0 True values
    1136 False values
nhạc có lời:
    ################# vocal1.wav
    63411 True values
    6952 False values
    ################# vocal2.wav
    75543 True values
    16800 False values
    ################# vocal3.wav
    9823 True values
    0 False values
    ################# nhactre.012.wav
    2408 True values
    7415 False values
    ################# nhactre.013.wav
    9823 True values
    0 False values
    ################# nhactre.011.wav
    9823 True values
    0 False values
    ################# nhactre.008.wav
    9761 True values
    62 False values
    ################# nhactre.016.wav
    9823 True values
    0 False values
    ################# nhactre.010.wav
    9823 True values
    0 False values
    ################# nhactre.014.wav
    6423 True values
    3400 False values
    ################# nhactre.015.wav
    9728 True values
    95 False values
    ################# nhactre.006.wav
    9168 True values
    655 False values
    ################# nhactre.009.wav
    9440 True values
    383 False values
    ################# nhactre.007.wav
    7838 True values
    1985 False values
    """
