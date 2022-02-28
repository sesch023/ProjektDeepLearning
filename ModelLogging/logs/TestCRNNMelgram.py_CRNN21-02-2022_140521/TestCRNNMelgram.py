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

with tf.device("/device:GPU:2"):
    wav_data_path = "../wav_data.dump"
    mfcc_dump = "../melgram_cnn.dump"
    test_size = 0.1
    n_mel = 96
    learning_rate = 0.001
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


    def MusicTaggerCRNN(input_shape=None):
        melgram_input = Input(shape=input_shape)

        TRAINABLE_HEAD=True

        print(melgram_input.shape)
        # Input block
        x = ZeroPadding2D(padding=(0, 37))(melgram_input)
        x = BatchNormalization(name='bn_0_freq')(x)

        print(x.shape)
        # Conv block 1
        x = Convolution2D(64, 3, 3, padding='same', name='conv1', trainable=TRAINABLE_HEAD)(x)
        x = BatchNormalization(name='bn1', trainable=TRAINABLE_HEAD)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool1', trainable=TRAINABLE_HEAD)(x)
        x = Dropout(0.1, name='dropout1', trainable=TRAINABLE_HEAD)(x)

        print(x.shape)
        # Conv block 2
        x = Convolution2D(128, 3, 3, padding='same', name='conv2', trainable=TRAINABLE_HEAD)(x)
        x = BatchNormalization(name='bn2', trainable=TRAINABLE_HEAD)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool2', trainable=TRAINABLE_HEAD)(x)
        x = Dropout(0.1, name='dropout2', trainable=TRAINABLE_HEAD)(x)

        print(x.shape)
        # Conv block 3
        x = Convolution2D(128, 3, 3, padding='same', name='conv3', trainable=TRAINABLE_HEAD)(x)
        x = BatchNormalization(name='bn3', trainable=TRAINABLE_HEAD)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool3', trainable=TRAINABLE_HEAD)(x)
        x = Dropout(0.1, name='dropout3', trainable=TRAINABLE_HEAD)(x)

        print(x.shape)
        # Conv block 4
        x = Convolution2D(128, 3, 3, padding='same', name='conv4', trainable=TRAINABLE_HEAD)(x)
        x = BatchNormalization(name='bn4', trainable=TRAINABLE_HEAD)(x)
        x = ELU()(x)
        print(x.shape)
        x = Dropout(0.1, name='dropout4', trainable=TRAINABLE_HEAD)(x)

        x = Reshape((8, 128))(x)

        # GRU block 1, 2, output
        x = GRU(32, return_sequences=True, name='gru1')(x)
        x = GRU(32, return_sequences=False, name='gru2')(x)
        x = Dropout(0.3, name='final_drop')(x)

        # Create model
        x = Dense(10, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
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
    max_len = 5280
    x = pad_sequences(x, value=mask_value, maxlen=max_len)
    y = np.array(y)
    x_val = pad_sequences(x_val, value=mask_value, maxlen=max_len)
    y_val = np.array(y_val)

    x_test = pad_sequences(x_test, value=mask_value, maxlen=max_len)
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
