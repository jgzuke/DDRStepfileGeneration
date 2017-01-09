import gc
import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from os import listdir
from sklearn.externals import joblib
from sklearn.utils import shuffle
from xgboost import XGBClassifier
np.random.seed(0)


# ### TODOS
# - generate percent of single double notes etc with a nn

# ## Classes
# - 0: one note
# - 1: two notes
# - 2: three or four notes
# - 3: hold start
# - 4: roll start
# - 5: mine
songs_to_use_full = pd.read_csv('training_data/songs_to_use.csv').values
save_files = listdir('training_data')
songs_to_use = [song_data for song_data in songs_to_use_full if '{0}_notes.csv'.format(song_data[0]) in save_files]
np.random.shuffle(songs_to_use)

def get_class_for_index_expanded(notes, index):
    if index < 0:
        return [0, 0, 0, 0, 0, 0]
    row = notes[index][0]
    (steps, holds, rolls, mines) = [row.count(char) for char in ['1', '2', '4', 'M']]
    if steps == 0 and mines == 0 and holds == 0 and rolls == 0:
        return [0, 0, 0, 0, 0, 0]
    steps += (holds + rolls)
    return [int(i) for i in [steps == 1, steps == 2, steps > 2, holds > 0, rolls > 0, mines > 0]]

def get_class_for_index(notes, index):
    classes_expanded = get_class_for_index_expanded(notes, index)
    return [i for i in range(6) if classes_expanded[i]]

class SongFile:
    def __init__(self, key):
        self.notes = pd.read_csv('training_data/{0}_notes.csv'.format(key), converters={'0': lambda x: str(x)}).values
        self.note_classes = [get_class_for_index_expanded(self.notes, i) for i in range(len(self.notes))]

note_types = ['1', 'M', '2', '4', '3']
def get_features_for_row(row):
    return [int(char == target) for target in note_types for char in row]

empty_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def get_previous_notes(index, features):
    previous_notes = [features[i] for i in range(index, index + song_padding) if not np.array_equal(features[i], empty_row)]
    return [empty_row] * (8 - len(previous_notes)) + previous_notes[-8:]

beats_to_track = 48
song_padding = beats_to_track * 2
song_end_padding = beats_to_track * 2
important_indices = [1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
important_indices_classes = [-96, -84, -72, -60, -48, -36, -24, -12, 0, 1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
def get_features(index, features, note_classes):
    indices = [index + song_padding - i for i in important_indices]
    indices_classes = [index + song_padding - i for i in important_indices_classes]
    past_classes = np.array([note_classes[i] for i in indices_classes]).flatten()
    past_features = np.array([features[i] for i in indices]).flatten()
    previous_notes = np.array(get_previous_notes(index, features)).flatten()
    return np.concatenate((past_classes, past_features, previous_notes), axis = 0)

def get_model_output_for_class(model_class, row):
    if model_class == 0 or model_class == 1 or model_class == 2:
        return [int(char == '1' or char == '2' or char == '4') for char in row]
    if model_class == 3:
        return [int(char == '2') for char in row]
    if model_class == 4:
        return [int(char == '4') for char in row]
    if model_class == 5:
        return [int(char == 'M') for char in row]

def get_hold_length(notes, note_row, note_column):
    i = 0
    while i < len(notes) - note_row:
        if notes[note_row + i][0][note_column] == '3':
            return i
        i += 1
    return False

def get_features_for_songs(songs):
    hold_X = []
    roll_X = []
    hold_y = []
    roll_y = []
    X = [[] for i in range(6)]
    y = [[] for i in range(6)]
    for song in songs:
        note_classes = np.concatenate((([[0, 0, 0, 0, 0, 0]] * song_padding), song.note_classes, ([[0, 0, 0, 0, 0, 0]] * song_end_padding)), axis = 0)
        notes = np.concatenate((([['0000']] * song_padding), song.notes), axis = 0)
        if abs(len(note_classes) - len(notes) > 250):
            print ('Lengths dont match for {0}'.format(key))
            print ('{0} vs {1}'.format(len(note_classes), len(notes)))
            continue
        length = min(len(note_classes) - song_padding - song_end_padding, len(notes) - song_padding)
        features = np.array([get_features_for_row(notes[i][0]) for i in range(0, length + song_padding)])
        for i in range(length):
            row = notes[i + song_padding][0]
            model_classes = get_class_for_index(notes, i + song_padding)
            for model_class in model_classes:
                X_row = get_features(i, features, note_classes)
                X[model_class].append(X_row)
                y[model_class].append(get_model_output_for_class(model_class, row))

                if model_class == 3:
                    for j in range(4):
                        if row[j] == '2':
                            length = get_hold_length(notes, i + song_padding, j)
                            if length:
                                hold_X.append(X_row)
                                hold_y.append(length)
                if model_class == 4:
                    for j in range(4):
                        if row[j] == '4':
                            length = get_hold_length(notes, i + song_padding, j)
                            if length:
                                roll_X.append(X_row)
                                roll_y.append(length)

    X = [np.array(X_for_class) for X_for_class in X]
    y = [np.array(y_for_class) for y_for_class in y]
    return X, y, np.array(hold_X), np.array(hold_y), np.array(roll_X), np.array(roll_y)

songs = [SongFile(song_data[0]) for song_data in songs_to_use]
X_array, y_array, hold_X, hold_y, roll_X, roll_y = get_features_for_songs(songs)

# Hold length model
model = Sequential()

model.add(Dense(512, input_dim=812))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adagrad')

model.fit(hold_X, hold_y, nb_epoch=8, batch_size=32, verbose=0)
model.save('models/hold_length_model.h5')

# Roll length model
model = Sequential()

model.add(Dense(256, input_dim=812))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))

model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adagrad')

model.fit(roll_X, roll_y, nb_epoch=40, batch_size=16, verbose=0)
model.save('models/roll_length_model.h5')

# Mapping of prediction classes to note outputs
class_arrays = [
    ['1000', '0100', '0010', '0001'],
    ['1100', '1010', '1001', '0110', '0101', '0011'],
    ['1110', '1101', '1011', '0111', '1111'],
    ['1000', '0100', '0010', '0001', '2', '3', '4'],
    ['1000', '0100', '0010', '0001', '2', '3', '4'],
    ['1000', '0100', '0010', '0001', '2', '3', '4'],
]
class_maps = [dict((class_array[i], i) for i in range(len(class_array))) for class_array in class_arrays]
def get_class(class_map, y_row):
    as_string = ''.join(str(x) for x in y_row)
    pos_count = as_string.count('1')
    return class_map[str(pos_count)] if '2' in class_map and pos_count > 1 else class_map[as_string]

def get_y_not_one_hot(y):
    return [[get_class(class_map, y_row) for y_row in y_section] for class_map, y_section in zip(class_maps, y)]

y_class_array = get_y_not_one_hot(y_array)


# Note Models
max_depths = [7, 9, 3, 6, 5, 5]
min_child_weights = [3, 3, 3, 3, 3, 3]
num_estimators = [120, 120, 60, 50, 75, 90]
for max_depth, min_child_weight, n_estimators, X, y, i in zip(max_depths, min_child_weights, num_estimators, X_array, y_class_array, range(6)):
    if len(X) == 0:
        print ('No examples of class to train on')
        continue
    xgb_clf = XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=0.1, n_estimators=n_estimators, subsample=0.70, colsample_bytree=0.70, objective="multi:softprob")
    xgb_clf.fit(X, y)
    print (xgb_clf.score(X, y))
    joblib.dump(xgb_clf, 'models/note_class_xgb/clf_{0}.pkl'.format(i))
