from __future__ import print_function
import argparse
import gc
import librosa
import numpy as np
import pandas as pd
import random
import re
from keras.models import load_model
from operator import itemgetter
from os import listdir, makedirs
from os.path import exists
from shutil import copyfile
from sklearn.externals import joblib


# Get beat features
steps_per_bar = 48
sample_rate_down = 1
hop_length_down = 8
sr = 11025 * 16 / sample_rate_down
hop_length = 512 / (sample_rate_down * hop_length_down)
samples_per_beat = steps_per_bar / 4

def load_misc_from_music(y):
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return (beat_times[0], get_beats(beat_times, beat_frames))

def get_beats(beat_times, beat_frames):
    changes = []
    changes_time = []
    for i in range(len(beat_frames) - 1):
        changes.append(beat_frames[i + 1] - beat_frames[i])
        changes_time.append(beat_times[i + 1] - beat_times[i])

    sorted_changes = sorted(changes)
    median = sorted_changes[int(len(changes) / 2)]
    median = max(set(sorted_changes), key=sorted_changes.count)

    changes_counted = [False] * len(changes)
    time_changes_sum = 0
    time_changes_count = 0
    for i in range(len(changes)):
        # can use other factors (eg if song has a slow part take double beats into accout)
        # in [0.5, 1, 2]:
        for change_factor in [1]:
            if abs((changes[i] * change_factor) - median) <= hop_length_down:
                changes_counted[i] = True
                time_changes_sum += (changes_time[i] * change_factor)
                time_changes_count += change_factor

    average = time_changes_sum / time_changes_count

    time_differences = []
    earliest_proper_beat = 1
    for i in range(1, len(beat_times) - 1):
        if changes_counted[i] & changes_counted[i - 1]:
            earliest_proper_beat = i
            break

    last_proper_beat = len(beat_times) -2
    for i in range(1, len(beat_times) - 1):
        if changes_counted[len(beat_times) - i - 1] & changes_counted[len(beat_times) - i - 2]:
            last_proper_beat = len(beat_times) - i - 1
            break

    time_differences = []
    buffer = 5
    for i in range(20):
        start_beat = earliest_proper_beat + buffer * i
        if changes_counted[start_beat] & changes_counted[start_beat - 1]:
            for j in range(20):
                end_beat = last_proper_beat - buffer * j
                if changes_counted[end_beat] & changes_counted[end_beat - 1]:
                    time_differences.append(beat_times[end_beat] - beat_times[start_beat])

    # get num beats, round, and make new average
    new_averages = [time_difference / round(time_difference / average) for time_difference in time_differences]
    new_averages.sort()
    num_averages = len(new_averages)
    new_average = new_averages[int(num_averages/2)]
    bpm = 60./new_average
    while bpm >= 200:
        bpm /= 2
    while bpm < 100:
        bpm *= 2
    # most songs have a few given bpms
    for target in [112, 118, 120, 124, 140, 148, 150, 156, 166, 176, 180, 200]:
        if abs(bpm - target) < 1:
            bpm = target

    return round(bpm)

def calculate_indices(offset, bpm, y):
    # take samples_per_beat samples for each beat (need 3rds, 8ths)
    seconds = len(y) / sr
    num_samples = int(seconds * samples_per_beat * bpm / 60)
    beat_length = 60. / bpm
    sample_length = beat_length / samples_per_beat

    if offset < 0:
        offset += 4 * beat_length

    sample_times = [offset + (sample_length * i) for i in range(num_samples)]
    # only take samples where music still playing
    indices = [round(time * sr) for time in sample_times if round(time * sr) < len(y)]
    # round down to nearest bar
    length = steps_per_bar * int(len(indices) / steps_per_bar) - 1
    return indices[:length]

def calculate_features(indices, y):
    y_harmonic = librosa.effects.harmonic(y)
    beat_frames = librosa.samples_to_frames(indices)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)
    y_harmonic = None
    y_percussive = None
    chromagram = None
    gc.collect()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
    mfcc = None
    mfcc_delta = None
    gc.collect()

    custom_hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=custom_hop)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env, hop_length=custom_hop)

    i = 0
    onset_happened_in_frame = [0] * (len(indices) + 1)
    for onset in onsets:
        onset_scaled = onset * custom_hop
        while i + 1 < len(indices) and abs(onset_scaled - indices[i]) > abs(onset_scaled - indices[i + 1]):
            i += 1
        onset_happened_in_frame[i] = max(onset_env[onset], onset_env[onset + 1], onset_env[onset + 2], onset_env[onset + 3], onset_env[onset + 4])

    zero_indexed_indices = [0] + indices
    max_offset_bounds = [(int(zero_indexed_indices[i] / custom_hop), int(zero_indexed_indices[i + 1] / custom_hop)) for i in range(len(zero_indexed_indices) - 1)]
    max_offset_strengths = [max(onset_env[bounds[0]:bounds[1]]) for bounds in max_offset_bounds]
    max_offset_strengths.append(0)

    return np.vstack([beat_chroma, beat_mfcc_delta, [onset_happened_in_frame, max_offset_strengths]])


# Get beat importance
samples_back_included_indices = [0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 36, 48]
samples_back_included = len(samples_back_included_indices)
num_features = 44

def get_features_for_index(beat_features, index):
    return beat_features[index] if index >= 0 else [0] * num_features

importance_rankings = [48, 24, 12, 6, 3, 16, 8, 4, 2, 1]
def get_beat_importance(index):
    for i in range(len(importance_rankings)):
        if index % importance_rankings[i] == 0:
            return i

def get_features_for_song(beat_features_rotated):
    beat_features = np.flipud(np.rot90(np.array(beat_features_rotated)))
    num_notes = len(beat_features)
    new_beat_features = [np.concatenate((beat_feature_row, [i % 48, get_beat_importance(i), i / 48, num_notes - i / 48]), axis=0) for beat_feature_row, i in zip(beat_features, range(len(beat_features)))]
    return np.array([[feature for j in samples_back_included_indices for feature in get_features_for_index(new_beat_features, i - j)] for i in range(num_notes)])


# Get song output
# Mapping of each class to new class given 0, 1... 4 holds
hold_class_redirect_array = [
    [],
    [1, 1, 0, 0, 0],
    [2, 1, 0, 0, 0],
    [3, 2, 1, 1, 0],
    [4, 4, 1, 0, 0],
    [5, 5, 0, 0, 0],
    [6, 0, 0, 0, 0]
]
# Mapping of model output to prediction (2/3 mean that many notes are present)
class_arrays = [
    [],
    ['1000', '0100', '0010', '0001'],
    ['1100', '1010', '0101', '0011', '1001', '0110'],
    ['1110', '1101', '1011', '0111', '1111'],
    ['2000', '0200', '0020', '0002', '2', '3', '2222'],
    ['4000', '0400', '0040', '0004', '2', '3', '4444'],
    ['M000', '0M00', '00M0', '000M', '2', '3', 'MMMM'],
]

beats_to_track = 48
note_types = ['1', 'M', '2', '4', '3']
def get_features_for_row(row):
    return [int(char == target) for target in note_types for char in row]

empty_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def get_previous_notes(index, features):
    previous_notes = [features[i] for i in range(index, index + song_padding) if not np.array_equal(features[i], empty_row)]
    return [empty_row] * (8 - len(previous_notes)) + previous_notes[-8:]

song_padding = beats_to_track * 2
song_end_padding = beats_to_track * 2
important_indices = [1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
important_indices_classes = [-96, -84, -72, -60, -48, -36, -24, -12, 0, 1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
def get_features(index, features, note_classes):
    indices = [index + song_padding - i for i in important_indices]
    indices_classes = [index + song_padding - i for i in important_indices_classes]
    past_classes = np.array([note_classes[i][1:] for i in indices_classes]).flatten()
    past_features = np.array([features[i] for i in indices]).flatten()
    previous_notes = np.array(get_previous_notes(index, features)).flatten()
    return np.concatenate((past_classes, past_features, previous_notes), axis = 0)

surrounding_beat_indices = [i for i in range(-24, 25)]#[-48, -36, -24, -12, 12, 24, 36, 48]
def get_average_for_class(surrounding_classes, class_num):
    return float(sum([beat_class[class_num] for beat_class in surrounding_classes])) / float(len(surrounding_classes))

def normalize_row(beat_class, surrounding_classes):
    return [beat_class[class_num] / get_average_for_class(surrounding_classes, class_num) for class_num in range(7)]

def normalize_classes(note_classes):
    return [normalize_row(note_classes[i], note_classes[max(0, i - 24):min(len(note_classes), i + 24)]) for i in range(len(note_classes))]

def replace_char(prediction, i, new_char):
    return prediction[:i] + new_char + prediction[i+1:]

hold_lengths = [3, 6, 8, 9, 12, 18, 24, 30, 36, 42, 48]
def get_closest_hold_length(length):
    return hold_lengths[np.argmax([-abs(length - aprox) for aprox in hold_lengths])]

pattern = ['1000', '0100', '0001', '0010', '0100', '1000', '0001', '0010', '1000', '0100', '0001', '0010', '0100', '1000', '0001', '0010']
default_class_cutoff_ammounts = [0.14, 0.03, 0, 0.02, 0, 0]
def get_output(note_classes, note_models, hold_length_model, roll_length_model, class_cutoff_ammounts):
    hold_lengths_current = [0, 0, 0, 0]
    roll_lengths_current = [0, 0, 0, 0]
    hold_lengths_max = [12, 12, 12, 12]
    roll_lengths_max = [12, 12, 12, 12]
    predicted_notes = []
    # TODO: figure out better normilazation
    normalized_note_classes = normalize_classes(note_classes)
    num_samples = len(note_classes)

    # get amt of each type of note
    if class_cutoff_ammounts == None:
        class_cutoff_ammounts = default_class_cutoff_ammounts

    class_cutoffs = [sorted(normalized_note_classes, key=itemgetter(i))[-max(int(num_samples * class_cutoff_ammounts[i - 1]), 1)][i] for i in range(1, 7)]

    note_classes = np.concatenate((([[1, 0, 0, 0, 0, 0, 0]] * song_padding), note_classes, ([[1, 0, 0, 0, 0, 0, 0]] * song_end_padding)), axis = 0)
    dummy_rows = [row for eigth in pattern for row in [eigth] + ['0000'] * 5]
    features = [get_features_for_row(row) for row in dummy_rows]
    for i in range(num_samples):
        note_class = note_classes[i]
        normalized_note_class = normalized_note_classes[i]
        prediction = '0000'
        X_row = get_features(len(features) - song_padding, features, note_classes)
        # order by reverse importance of decision
        # TODO up limit if something has been bumped out of existance (eg put more jumps if they get covered by holds)
        targets = ['0', '1', '1', '1', '2', '4', 'M']
        ammounts = [0, 1, 2, 3, 1, 1, 1]
        for i in [1, 6, 2, 5, 4, 3]:
            if normalized_note_class[i] > class_cutoffs[i - 1]:
                holds = sum(length > 0 for length in hold_lengths_current) + sum(length > 0 for length in roll_lengths_current)
                new_class = hold_class_redirect_array[i][holds]
                if new_class == 0:
                    continue
                prediction_class = note_models[new_class].predict(np.array([X_row]))[0]
                # mix things up
                if random.randint(0, 3) == 0:
                    prediction = class_arrays[new_class][prediction_class]
                else:
                    prediction = class_arrays[new_class][random.randint(0, 3)]

                # replace 2, 3 for mines, rolls, holds
                if len(prediction) == 1:
                    num = len(prediction)
                    # take [:4] because only prop for each of first 4 classes matters
                    prediction_values = note_models[new_class].predict_proba(np.array([X_row]))[0][:4]
                    cutoff = sorted(prediction_values)[-(num + 1)]
                    prediction = ''.join([targets[new_class] if value > cutoff else '0' for value in prediction_values])

        for i in range(4):
            if hold_lengths_current[i] > 0:
                hold_lengths_current[i] += 1
                if hold_lengths_current[i] == hold_lengths_max[i]:
                    prediction = replace_char(prediction, i, '3')
                    hold_lengths_current[i] = 0
                else:
                    prediction = replace_char(prediction, i, '0')
            if roll_lengths_current[i] > 0:
                roll_lengths_current[i] += 1
                if roll_lengths_current[i] == roll_lengths_max[i]:
                    prediction = replace_char(prediction, i, '3')
                    roll_lengths_current[i] = 0
                else:
                    prediction = replace_char(prediction, i, '0')
            if prediction[i] == '2':
                hold_lengths_current[i] = 1
                hold_lengths_max[i] = get_closest_hold_length(hold_length_model.predict(np.array([X_row]))[0][0])
            if prediction[i] == '4':
                roll_lengths_current[i] = 1
                roll_lengths_max[i] = get_closest_hold_length(roll_length_model.predict(np.array([X_row]))[0][0])

        predicted_notes.append(prediction)
        features.append(get_features_for_row(prediction))
    return predicted_notes


# Write file
def write_song_metadata(output_stepfile, song, music_file, offset, bpm):
    keys = ['TITLE', 'MUSIC', 'OFFSET', 'SAMPLESTART', 'SAMPLELENGTH', 'SELECTABLE', 'BPMS']
    header_info = {
        'TITLE': song,
        'MUSIC': music_file,
        'OFFSET': -offset,
        'SAMPLESTART': offset + 32 * (60. / bpm),
        'SAMPLELENGTH': 32 * (60. / bpm),
        'SELECTABLE': 'YES',
        'BPMS': '0.000={:.3f}'.format(bpm)
    }

    for key in keys:
        print ("#{0}:{1};".format(key, str(header_info[key])), file=output_stepfile)

def write_song_steps(output_stepfile, predicted_notes):
    print("\n//---------------dance-single - J. Zukewich----------------", file=output_stepfile)
    print ("#NOTES:", file=output_stepfile)
    for detail in ['dance-single', 'J. Zukewich', 'Expert', '9', '0.242,0.312,0.204,0.000,0.000']:
        print ('\t{0}:'.format(detail), file=output_stepfile)

    for i in range(len(predicted_notes)):
        row = predicted_notes[i]
        print (row, file=output_stepfile)
        if i == len(predicted_notes) - 1:
            print (';', file=output_stepfile)
        if i % steps_per_bar == steps_per_bar - 1 and i != len(predicted_notes) - 1:
            print (",", file=output_stepfile)


# Step songs
def step_songs(music_files, regenerate_features, regenerate_note_classes, regenerate_notes, class_cutoff_ammounts):
    song_class_model = None
    song_class_scaler = None
    hold_length_model = None
    roll_length_model = None
    note_models = None

    for music_file in music_files:
        song, _ = music_file.split('.')
        key = song
        folder = '../{0}/'.format(song)
        stepfile_name = '{0}.sm'.format(song)
        saved_data = listdir('stepping_data')
        if not exists(folder):
            makedirs(folder)
        copyfile('stepping_songs/' + music_file, folder + music_file)

        if not regenerate_features and ('{0}_beat_features.csv'.format(key) in saved_data and '{0}_misc.csv'.format(key) in saved_data):
            print ('Loadind Saved Features for {0}'.format(song))
            [offset], [bpm] = pd.read_csv('stepping_data/{0}_misc.csv'.format(key)).values
            beat_features = pd.read_csv('stepping_data/{0}_beat_features.csv'.format(key)).values
        else:
            print ('Loading Song {0}'.format(song))
            y, _ = librosa.load('stepping_songs/' + music_file, sr=sr)

            print ('Calculating BPM')
            offset, bpm = load_misc_from_music(y)
            pd.DataFrame([offset, bpm]).to_csv('stepping_data/{0}_misc.csv'.format(key), index=False)

            print ('Calculating Features')
            indices = calculate_indices(offset, bpm, y)
            beat_features = calculate_features(indices, y)
            pd.DataFrame(beat_features).to_csv('stepping_data/{0}_beat_features.csv'.format(key), index=False)
        y = None
        indices = None

        if not regenerate_note_classes and ('{0}_note_classes_generated.csv'.format(key) in saved_data):
            print ('Loading Song Predicted Classes')
            note_classes = pd.read_csv('stepping_data/{0}_note_classes_generated.csv'.format(key)).values
        else:
            print ('Getting Song Predicted Classes')
            if song_class_model == None:
                song_class_model = load_model('models/song_class_model.h5')
                song_class_scaler = joblib.load('models/song_class_scaler/scaler.pkl')

            X = get_features_for_song(beat_features)
            X = song_class_scaler.transform(X)
            X = np.reshape(X, (X.shape[0], samples_back_included, num_features))

            note_classes = song_class_model.predict(X[:int(len(X) / 96) * 96], batch_size=96)
            pd.DataFrame(note_classes).to_csv('stepping_data/{0}_note_classes_generated.csv'.format(key), index=False)
        beat_features = None
        X = None

        if not regenerate_notes and ('{0}_predicted_notes.csv'.format(key) in saved_data):
            print ('Loading Predicted Notes')
            predicted_notes = pd.read_csv('stepping_data/{0}_predicted_notes.csv'.format(key)).values
        else:
            print ('Predicting Notes')
            if note_models == None:
                hold_length_model = load_model('models/hold_length_model.h5')
                roll_length_model = load_model('models/roll_length_model.h5')
                note_models = [None] + [joblib.load('models/note_class_xgb/clf_{0}.pkl'.format(i)) for i in range(6)]

            predicted_notes = get_output(note_classes, note_models, hold_length_model, roll_length_model, class_cutoff_ammounts)
            pd.DataFrame(predicted_notes).to_csv('stepping_data/{0}_predicted_notes.csv'.format(key), index=False)
        note_classes = None

        print ('Writing Song to File')
        stepfile=open(folder + stepfile_name, 'w')
        write_song_metadata(stepfile, song, music_file, offset, bpm)
        write_song_steps(stepfile, predicted_notes)
        stepfile.close()

        print ('Done')


parser = argparse.ArgumentParser(description='Step Generation')
parser.add_argument('--songs', nargs='+', default=None,
                    help='songs to step')
parser.add_argument('--regenerate_features', dest='regenerate_features', action='store_const',
                    const=True, default=False,
                    help='Force the song to reload and generate new features from music file')
parser.add_argument('--regenerate_note_classes', dest='regenerate_note_classes', action='store_const',
                    const=True, default=False,
                    help='Force the song to regenerate notes')
parser.add_argument('--regenerate_notes', dest='regenerate_notes', action='store_const',
                    const=True, default=False,
                    help='Force the song to reload and generate new features from music file')
parser.add_argument("--class_ammounts", nargs=6, type=float, default=None,
                    help='Fraction of notes of each class to generate [step, jump, hand, hold, roll, mine]')

args = parser.parse_args()

if args.songs:
    songs = args.songs
else:
    songs = [song for song in listdir('stepping_songs') if song.split('.')[1] in ['ogg', 'mp3', 'mp4', 'wav']]

regenerate_features = args.regenerate_features
regenerate_note_classes = args.regenerate_note_classes or regenerate_features
regenerate_notes = args.regenerate_notes or regenerate_note_classes
step_songs(songs, regenerate_features, regenerate_note_classes, regenerate_notes, args.class_ammounts)
