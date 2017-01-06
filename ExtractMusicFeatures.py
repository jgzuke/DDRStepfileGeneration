import gc
import librosa
import numpy as np
import pandas as pd
import re
from os import listdir

# # SongFile:
# ### Fields
# - beat_frames:
# - beat_times:
# - bpm:
# - bpm_string:
# - beat_length:
# - indices:
# - data:
#
# - pack
# - name
# - extension
# - music_file
# - stepfile
# ### Output
# - data/{0}_beat_features.csv
# - data/{0}_misc.csv

steps_per_bar = 48
sample_rate_down = 1
hop_length_down = 8
sr = 11025 * 16 / sample_rate_down
hop_length = 512 / (sample_rate_down * hop_length_down)
samples_per_beat = steps_per_bar / 4

def load_misc_from_stepfile(stepfile):
    with open(stepfile, "r") as txt:
        step_file = txt.read()
        step_file = step_file.replace('\n', 'n')
        bpm_search = re.search('#BPMS:([0-9.=,]*);', step_file)
        bpm_string = bpm_search.group(1)
        bpm = float(bpm_string.split('=')[1]) if len(bpm_string.split(',')) == 1 else 0

        offset_search = re.search('#OFFSET:([0-9.-]*);', step_file)
        offset = -float(offset_search.group(1))
        return (offset, bpm)

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

songs_to_use = pd.read_csv('training_data/songs_to_use.csv').values
for song_data in songs_to_use:
    if '{0}_beat_features.csv'.format(song_data[0]) in listdir('training_data'):
        print ('Song Already Loaded')
    else:
        try:
            y, _ = librosa.load(song_data[3], sr=sr)

            print ('Calculating BPM')
            offset, bpm = load_misc_from_stepfile(song_data[2])
            pd.DataFrame([offset, bpm]).to_csv('training_data/{0}_misc.csv'.format(song_data[0]), index=False)

            print ('Calculating Features')
            indices = calculate_indices(offset, bpm, y)
            beat_features = calculate_features(indices, y)
            pd.DataFrame(beat_features).to_csv('training_data/{0}_beat_features.csv'.format(song_data[0]), index=False)
        except:
            print ('Error loading {0}\n'.format(song_data[0]))
        gc.collect()
