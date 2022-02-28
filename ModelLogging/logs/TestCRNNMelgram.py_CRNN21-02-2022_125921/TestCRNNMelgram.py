import sys

from tensorflow.keras.layers import MaxPooling2D, ELU, Convolution2D, Permute, Reshape, ZeroPadding2D

sys.path.append('/home/sebastian/ProjektDeepLearning') # Hack

import pickle
from collections import namedtuple
from Utils.Util import Song, evaluate
import librosa.feature
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Masking, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ModelLogging.LogRun import log_run
from tensorflow.keras.models import load_model

wav_data_path = "../wav_data.dump"
mfcc_dump = "../melgram.dump"
test_size = 0.1
n_mel = 96
learning_rate = 0.0001
batch_size = 32
reset = False
name_extension = "CRNN"


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


def extract_melgram_single(data, sr, n_mels=96):
    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=data, sr=sr, hop_length=256,
                        n_fft=512, n_mels=n_mels) ** 2,
                ref=1.0)
    return ret


def extract_melgram(data, sr, n_mels=96):
    print("Extracting Melgram")
    return [extract_melgram_single(element, sre, n_mels) for element, sre in tqdm(zip(data, sr))]


def reformat(data):
    return [np.array(x_ele).swapaxes(0, 1)[:, :, np.newaxis] for x_ele in data]


def MusicTaggerCRNN(weights='msd', input_tensor=None, input_shape=None):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    # Determine input axis
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, padding='same', name='conv1', trainable=False)(x)
    x = BatchNormalization(name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1', trainable=False)(x)
    x = Dropout(0.1, name='dropout1', trainable=False)(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, padding='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2', trainable=False)(x)
    x = Dropout(0.1, name='dropout2', trainable=False)(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, padding='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3', trainable=False)(x)
    x = Dropout(0.1, name='dropout3', trainable=False)(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, padding='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool4', trainable=False)(x)
    x = Dropout(0.1, name='dropout4', trainable=False)(x)

    # reshaping
    if K.image_data_format() == 'channels_first':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
        x = Dense(10, activation='sigmoid',name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_data_format() == 'channels_last':
            raise RuntimeError("Please set image_dim_ordering == 'channels_last'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)

        # Eliminate last layer
        initial_model.layers.pop()

        # Add new Dense layer
        last = initial_model.get_layer('final_drop')
        preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        return model


if not os.path.isfile(mfcc_dump) or reset:
    x, y, sr = parse_wav_data(get_wav_data(wav_data_path))
    num_classes = len(set(y))
    y = LabelEncoder().fit_transform(y)
    x = reformat(extract_melgram(x, sr, n_mel))
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

model = MusicTaggerCRNN(input_shape=(x.shape[1], x.shape[2], 1))

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
