#!/usr/bin/env python
# coding: utf-8

import avgn
import pandas as pd
import xlrd
from datetime import timedelta
import os
import urllib.request
from tqdm import tqdm
from urllib.error import HTTPError
from sklearn.externals.joblib import Parallel, delayed

### read bird excel file
module_dir = os.path.abspath(f'{os.path.dirname(avgn.__file__)}/..')
song_db = pd.read_excel(f'{module_dir}/BIRD_DB.xls')
mainData_book = xlrd.open_workbook(f"{module_dir}/BIRD_DB.xls", formatting_info=True)
mainData_sheet = mainData_book.sheet_by_index(0)
song_urls = [
    '' if mainData_sheet.hyperlink_map.get((i, 11)) == None else mainData_sheet.hyperlink_map.get((i, 11)).url_or_path
    for i in range(mainData_sheet.nrows)]
song_db['Audio_file'] = song_urls[1:]
song_db = song_db[1:]

print(len(song_db), mainData_sheet.nrows)

### Download waveforms and TextGrids

bird_db_loc = f'{os.path.expanduser("~")}/Data/bird-db'
if not os.path.exists(bird_db_loc):
    os.makedirs(bird_db_loc)

parallel = False
verbosity = 0
n_jobs = 20
save = True


def downloadBirdDB(row):
    wav = row['Audio_file']
    text_grid = row['Textgrid_file']
    track_name = row['TrackName']
    subject_id = row['SubjectName']
    species = row['Species_short_name']
    recording_time = row['recording_date'] + timedelta(hours=row['recording_time'].hour,
                                                       minutes=row['recording_time'].minute,
                                                       seconds=row['recording_time'].second)
    # PREP SAVE LOCATION
    wav_location = '/'.join(
        [bird_db_loc, species, subject_id, 'wavs', recording_time.strftime("%Y-%m-%d_%H-%M-%S-%f") + '.wav'])
    grid_location = '/'.join(
        [bird_db_loc, species, subject_id, 'TextGrids', recording_time.strftime("%Y-%m-%d_%H-%M-%S-%f") + '.TextGrid'])
    if not os.path.exists('/'.join([bird_db_loc, species, subject_id, 'wavs'])):
        os.makedirs('/'.join([bird_db_loc, species, subject_id, 'wavs']))
    if not os.path.exists('/'.join([bird_db_loc, species, subject_id, 'TextGrids'])):
        os.makedirs('/'.join([bird_db_loc, species, subject_id, 'TextGrids']))

        # save wav
    if not os.path.exists(wav_location):
        try:
            urllib.request.urlretrieve(wav, wav_location)
        except HTTPError:
            print('Could not retreive ' + wav)
    # save textgrid
    if not os.path.exists(grid_location):
        try:
            urllib.request.urlretrieve('http://taylor0.biology.ucla.edu/birdDBQuery/Files/' + text_grid, grid_location)
        except HTTPError:
            print('Could not retreive ' + 'http://taylor0.biology.ucla.edu/birdDBQuery/Files/' + text_grid)


if parallel:
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        parallel(delayed(downloadBirdDB)(row)
                 for idx, row in tqdm(song_db.iterrows(), total=len(song_db)))
else:
    for idx, row in tqdm(song_db.iterrows(), total=len(song_db)):
        downloadBirdDB(row)
