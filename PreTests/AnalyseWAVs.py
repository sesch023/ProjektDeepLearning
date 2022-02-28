import glob
from pathlib import PurePath
from collections import namedtuple
import librosa
import os
from tqdm import tqdm
import traceback
import sys
import pickle
from Utils.Util import Song


DEFAULT_SAMPLE_RATE = 44100
MAX_SAMPLE_RATE = 100000
MIN_SAMPLE_RATE = 100


def _get_wavs(root_path, limit_wavs=None):
    if not os.path.isdir(root_path) or not os.access(root_path, os.R_OK):
        raise ValueError("_get_wavs: Path does not exist or is not readable!")

    data = list(glob.iglob(root_path + '**/*.wav', recursive=True))

    if limit_wavs is not None and (not isinstance(int, limit_wavs or not(0 < limit_wavs < len(data)))):
        raise ValueError("_get_wavs: Illegal Value for Wav-Read-Limit!")

    return data if limit_wavs is None else data[:limit_wavs]


def _read_wavs(wavs_paths, sample_rate=DEFAULT_SAMPLE_RATE, ignore_failed=True):
    if not isinstance(wavs_paths, list):
        raise ValueError("_read_wavs: Given wavs_paths is not a list!")

    if not isinstance(sample_rate, int) or not(MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE):
        raise ValueError("_read_wavs: Given sample_rate has illegal value!")

    data = []

    for wav in tqdm(wavs_paths):
        if not os.path.isfile(wav) or not os.access(wav, os.R_OK):
            if not ignore_failed:
                raise IOError(f"_read_wavs: \"{wav}\" is no file or is not accessible!")
            else:
                print(f"_read_wavs: \"{wav}\" is no file or is not accessible!", file=sys.stderr)
        try:
            song_data, sr = librosa.load(wav, sr=sample_rate)
            path_parts = (PurePath(wav.replace("wav", ""))).parts
            data.append(Song(name=path_parts[-1], genre=path_parts[-2], path=wav, data=song_data, sample_rate=sr))
        except Exception:
            traceback.print_exc()
            if not ignore_failed:
                raise IOError(f"_read_wavs: Error while reading File \"{wav}\"!")
            else:
                print(f"_read_wavs: Error while reading File \"{wav}\"!", file=sys.stderr)
    return data


def read_wavs(root_path, sample_rate=DEFAULT_SAMPLE_RATE, limit_wavs=None):
    return _read_wavs(_get_wavs(root_path, limit_wavs=limit_wavs), sample_rate=sample_rate)


data = read_wavs("../data/GTZAN Dataset/")
target_pickle = "../wav_data.dump"

if not os.path.isfile(target_pickle):
    with open(target_pickle, "wb") as file:
        pickle.dump(data, file)


