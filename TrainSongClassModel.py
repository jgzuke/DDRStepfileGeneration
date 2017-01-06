import gc
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM
from os import listdir
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
np.random.seed(0)


# # Note types
# - 0: nothing
# - 1: step
# - 2: hold start
# - 3: hold/roll end
# - 4: roll start
# - M: mine
#
# # Classes
# - 0: nothing
# - 1: one note
# - 2: two notes
# - 3: three or four notes
# - 4: hold start
# - 5: roll start
# - 6: mine

samples_back_included_indices = [0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 36, 48]
samples_back_included = len(samples_back_included_indices)
num_classes = 7
num_features = 44
num_features_total = (num_features * samples_back_included)
save_files = listdir('training_data')

def get_features_for_index(beat_features, index):
    return beat_features[index] if index >= 0 else [0] * num_features

def get_class_for_index_expanded(notes, index):
    if index < 0:
        return [1, 0, 0, 0, 0, 0, 0]
    row = notes[index][0]
    (steps, holds, rolls, mines) = [row.count(char) for char in ['1', '2', '4', 'M']]
    if steps == 0 and mines == 0 and holds == 0 and rolls == 0:
        return [1, 0, 0, 0, 0, 0, 0]
    steps += (holds + rolls)
    return [int(i) for i in [False, steps == 1, steps == 2, steps > 2, holds > 0, rolls > 0, mines > 0]]

def get_class_for_index(notes, index):
    classes_expanded = get_class_for_index_expanded(notes, index)
    return [i for i in range(7) if classes_expanded[i]]

importance_rankings = [48, 24, 12, 6, 3, 16, 8, 4, 2, 1]
def get_beat_importance(index):
    for i in range(len(importance_rankings)):
        if index % importance_rankings[i] == 0:
            return i

def get_features_for_song(X, y, key):
    if '{0}_beat_features.csv'.format(key) in save_files and '{0}_notes.csv'.format(key) in save_files:
        beat_features_rotated = pd.read_csv('training_data/{0}_beat_features.csv'.format(key)).values
        notes = pd.read_csv('training_data/{0}_notes.csv'.format(key), converters={'0': lambda x: str(x)}).values
        beat_features = np.flipud(np.rot90(np.array(beat_features_rotated)))
        num_notes = min(len(notes), len(beat_features))
        new_beat_features = []
        for beat_feature_row, i in zip(beat_features, range(len(beat_features))):
            new_beat_feature_row = np.concatenate((beat_feature_row, [i % 48, get_beat_importance(i), i / 48, num_notes - i / 48]), axis=0)
            new_beat_features.append(new_beat_feature_row)

        for i in range(num_notes):
            class_num = get_class_for_index_expanded(notes, i)
            features = [feature for j in samples_back_included_indices for feature in get_features_for_index(new_beat_features, i - j)]
            X.append(features)
            y.append(class_num)

def build_training_data(songs):
    X = []
    y = []
    for song_data in songs:
        get_features_for_song(X, y, song_data[0])
    return np.array(X), np.array(y)


# Full Model
songs_to_use = pd.read_csv('training_data/songs_to_use.csv').values
X, y = build_training_data(songs_to_use)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
joblib.dump(scaler, 'models/song_class_scaler/scaler.pkl')

X = np.reshape(X, (X.shape[0], samples_back_included, num_features))
gc.collect()

batch_size = 96

model = Sequential()
model.add(LSTM(128, batch_input_shape=[batch_size, samples_back_included, num_features], stateful=True))
model.add(BatchNormalization())
model.add(Activation('softsign'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

cutoff = int(len(X) / batch_size) * batch_size
model.fit(X[:cutoff], y[:cutoff], nb_epoch=8, batch_size=batch_size, verbose=1)
model.save('models/song_class_model.h5')
