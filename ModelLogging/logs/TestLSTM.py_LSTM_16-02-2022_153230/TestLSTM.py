import sys
sys.path.append('/home/sebastian/ProjektDeepLearning') # Hack

import pickle
from collections import namedtuple
from Utils.Util import Song, evaluate
import librosa.feature
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ModelLogging.LogRun import log_run
from tensorflow.keras.models import load_model

wav_data_path = "../wav_data.dump"
mfcc_dump = "../mfcc.dump"
test_size = 0.1
n_mfcc = 40
learning_rate = 0.00009
batch_size = 32
reset = False
name_extension = "LSTM_"


def get_wav_data(wav_data_path):
    with open(wav_data_path, "rb") as file:
        return pickle.load(file)


def parse_wav_data(wav_data):
    x = []
    y = []
    sr = []

    print("Parsing WAVs")
    for element in tqdm(wav_data):  # type: Song
        x.append(element.data)
        y.append(element.genre)
        sr.append(element.sample_rate)

    return x, y, sr


def extract_mfcc(data, sr, num_mfcc=20):
    print("Extracting MFCC")
    return [librosa.feature.mfcc(y=element, sr=sre, n_mfcc=num_mfcc) for element, sre in tqdm(zip(data, sr))]


def reformat_mfcc(data):
    return [np.array(x_ele).swapaxes(0, 1) for x_ele in data]


if not os.path.isfile(mfcc_dump) or reset:
    x, y, sr = parse_wav_data(get_wav_data(wav_data_path))
    num_classes = len(set(y))
    y = LabelEncoder().fit_transform(y)
    x = reformat_mfcc(extract_mfcc(x, sr, n_mfcc))
    with open(mfcc_dump, "wb") as file:
        pickle.dump((x, y, num_classes), file)
else:
    with open(mfcc_dump, "rb") as file:
        x, y, num_classes = pickle.load(file)

x, x_test, y, y_test = train_test_split(x, y, test_size=2*test_size)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
mask_value = -1000
x = pad_sequences(x, value=mask_value)
y = np.array(y)
x_val = pad_sequences(x_val, value=mask_value)
y_val = np.array(y_val)

x_test = pad_sequences(x_test, value=mask_value)
y_test = np.array(y_test)

tensorboard_callback, checkpoint, run_path = log_run("../ModelLogging/logs/", log_to_file=True,
                                                     script_name_extension=name_extension)

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(x.shape[1], x.shape[2])))
model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.25))
model.add(BatchNormalization())
model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.25))
model.add(BatchNormalization())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.fit(x, y,
          epochs=100,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[es, checkpoint],
          batch_size=batch_size)

model = load_model(run_path + "model.mdl_wts.hdf5")


def eval_on_dataset(model, x, y):
    score = model.evaluate(x, y, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Prediction für den Test Datensatz
    pred = model.predict(x, verbose=1)
    # Evaluiere die Ergebnisse vom Testdatensatz mit sklearn
    print(evaluate(np.argmax(pred, axis=1), y))


print("Valset")
eval_on_dataset(model, x_val, y_val)
print("Testset")
eval_on_dataset(model, x_test, y_test)
