import itertools
import pandas as pd
import re
from os import listdir

song_data = []
packs = [pack for pack in listdir('../../') if pack != 'Generated' and pack.count('.') == 0]
for pack in packs:
    songs = [song for song in listdir('../../{0}'.format(pack)) if song.count('.') == 0]
    for song in songs:
        try:
            key = '{0}~{1}'.format(pack, song)
            folder = '../../{0}/{1}/'.format(pack, song)
            stepfiles = [file for file in listdir(folder) if file.split('.')[1] in ['ssc', 'sm']]
            musicfiles = [file for file in listdir(folder) if file.split('.')[1] in ['ogg', 'mp3', 'mp4', 'wav']]
            if len(stepfiles) != 1 or len(musicfiles) != 1:
                continue

            stepfile = folder + stepfiles[0]
            music = folder + musicfiles[0]

            bpm = 0
            offset = 0

            with open(stepfile, "r") as txt:
                step_info = txt.read()
                step_info = step_info.replace('\n', 'n')
                bpm_search = re.search('#BPMS:([0-9.=,]*);', step_info)
                bpm_string = bpm_search.group(1)
                bpm = float(bpm_string.split('=')[1]) if len(bpm_string.split(',')) == 1 else 0

                offset_search = re.search('#OFFSET:([0-9.-]*);', step_info)
                offset = -float(offset_search.group(1))

            if bpm == 0 or offset < 0:
                continue

            song_data.append([key, folder, stepfile, music])
        except:
            print ('Error loading song {0}~{1}\n'.format(pack, song))

song_data_df = pd.DataFrame(song_data, columns=['KEY', 'FOLDER', 'STEPFILE', 'MUSIC'])
song_data_df.to_csv('training_data/songs_to_use.csv', index=False)
