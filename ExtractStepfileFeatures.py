import argparse
import pandas as pd
import re
from os import listdir


# # Helpers to read .sm and return notes and meta data
# - get_notes_from_note_string(note_string)
# - get_notes_and_metadata(file)
# - get_song_steps()

regex_notes_with_metadata = '#NOTES:n     dance-single((?:(?!//-).)*);'
regex_metadata_split = ':n     (.*):n     (.*):n     (.*):n     (.*):n     (.*):(.*);'
def get_notes_from_note_string(note_string):
    note_strings_split = re.split(r'n', note_string)[1:-1]
    notes = []
    bar = []
    for row in note_strings_split:
        if len(row) == 4:
            bar.append(row)
        else:
            notes.append(bar)
            bar = []
    return notes

def get_notes_and_metadata(file):
    difficulty_map = {}
    with open(file) as txt:
        step_file = txt.read()
        step_file = step_file.replace('\n', 'n')
        notes_with_metadata_groups = re.finditer(regex_notes_with_metadata, step_file)
        for match in notes_with_metadata_groups:
            notes_with_metadata = match.group(0)
            split_data = re.search(regex_metadata_split, notes_with_metadata)
            difficulty = split_data.group(4)
            metadata = split_data.group(5)
            notes = get_notes_from_note_string(split_data.group(6))
            notes_with_metadata_map = {
                'DIFFICULTY': difficulty,
                'METADATA': metadata,
                'NOTES': notes,
            }
            difficulty_map[difficulty] = notes_with_metadata_map
    return difficulty_map

notes_per_bar = 48
def padBar(bar):
    pad = int(48 / len(bar)) if len(bar) != 0 else 1
    return [row for note in bar for row in [note] + (pad - 1) * ['0000']]

def get_plain_padded_notes_from_note_string(stepfile, target_difficulty):
    notes_and_metadata = get_notes_and_metadata(stepfile)
    closest_difficulty = min(notes_and_metadata.values(), key=lambda steps:abs(int(steps['DIFFICULTY']) - target_difficulty))
    if abs(int(closest_difficulty['DIFFICULTY']) - target_difficulty) > 1:
        return None
    return [row for bar in closest_difficulty['NOTES'] for row in padBar(bar)]


parser = argparse.ArgumentParser(description='Extract Stepfile Features')
parser.add_argument('--target_difficulty', type=int, required=True,
                    help='diffulty to train model to predict')

args = parser.parse_args()


songs_to_use = pd.read_csv('training_data/songs_to_use.csv').values
for song_data in songs_to_use:
    if '{0}_notes.csv'.format(song_data[0]) in listdir('training_data'):
        print ('Stepfile Already Loaded')
    else:
        notes = get_plain_padded_notes_from_note_string(song_data[2], args.target_difficulty)
        if notes:
            pd.DataFrame(notes).to_csv('training_data/{0}_notes.csv'.format(song_data[0]), index=False)
        else:
            print ('No close difficulty for {0}\n'.format(song_data[0]))
