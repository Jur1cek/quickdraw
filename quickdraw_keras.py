import numpy as np
import keras
import keras.preprocessing.sequence
import sklearn.utils
import tensorflow as tf

from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Masking
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D, MaxPooling2D, AveragePooling2D

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_X = np.load("X.npy")
data_Y = np.load("Y.npy")

data_X, data_Y = shuffle(data_X, data_Y)

data_X_padded = keras.preprocessing.sequence.pad_sequences(data_X, maxlen=100, dtype='float32', padding='post', truncating='post')
data_X_padded = data_X_padded.reshape(data_X_padded.shape[0], data_X_padded.shape[1], data_X_padded.shape[2], 1)

le = preprocessing.LabelEncoder()
targets_labels = le.fit_transform(data_Y)
print(le.classes_)
one_hot_labels = keras.utils.to_categorical(targets_labels, num_classes=8)

print(data_X_padded.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), padding='same', activation='relu', input_shape=data_X_padded.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=96, kernel_size=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Bidirectional(LSTM(48, return_sequences=True))))
# model.add(Dropout(0.1))
model.add(TimeDistributed(Bidirectional(LSTM(32, return_sequences=True))))
# model.add(Dropout(0.1))
model.add(TimeDistributed(Bidirectional(LSTM(48))))
model.add(Flatten())
model.add(Dense(8, activation='softmax'))

parallel_model = model
parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = parallel_model.fit(data_X_padded, one_hot_labels, epochs=200, batch_size=1024, validation_split=0.1, verbose=2)

model.save("quickdraw.h5")
