from collections import namedtuple
import numpy as np
from sklearn import metrics

Song = namedtuple("Song", "name genre path data sample_rate")
SongMidi = namedtuple("SongMidi", "name artist genre subgenre lowest_note notes durations")


def evaluate(predicted, pred_labels, digits=2):
    """
    Evaluiere einen classifier auf Evaluierungsdaten.
    """
    print(f"Confusion matrix:\n{metrics.confusion_matrix(pred_labels, predicted)}")
    print(f"{metrics.classification_report(pred_labels, predicted, digits=digits)}")
    return np.mean(predicted == pred_labels)
