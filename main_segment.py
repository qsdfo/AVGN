#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import warnings
import wave
from datetime import datetime
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from praatio import tgio
from sklearn.externals.joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')
import avgn.spectrogramming.spectrogramming as sg
import avgn.segment_song.wav_to_syllables as w2s


dataset_name = 'CATH'

def norm_zero_one(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def read_grid(grid):
    tg = tgio.openTextgrid(grid)
    MotifTier = tg.tierDict['Motifs']
    return np.array(MotifTier.entryList).T


dt = h5py.special_dtype(vlen=str)


def save_dataset(location, all_bird_syll, starting_times, lengths, wav_file, syll_start_rel_to_wav, symbols, bird_name):
    with h5py.File(location, 'w') as f:
        f.attrs['bird_name'] = bird_name
        dset_spec = f.create_dataset("spectrograms", np.shape(all_bird_syll), dtype='uint8', data=all_bird_syll)
        dset_start = f.create_dataset("start", data=np.array(starting_times).astype('S'))
        dset_wav_file = f.create_dataset("wav_file", data=np.array(wav_file).astype('S'))
        dset_syll_start_rel_to_wav = f.create_dataset("syll_start_rel_wav",
                                                      np.shape(syll_start_rel_to_wav), dtype='float32',
                                                      data=syll_start_rel_to_wav)
        dset_symbols = f.create_dataset("symbols", data=np.array(symbols).astype('S'))
        dset_lengths = f.create_dataset("lengths", np.shape(lengths), dtype='float32', data=lengths)


syll_size = 128
hparams = {
    'species': dataset_name,
    # filtering
    'highcut': 20000,
    'lowcut': 500,

    # spectrograms
    'mel_filter': True,  # should a mel filter be used?
    'num_mels': syll_size,  # how many channels to use in the mel-spectrogram
    'num_freq': 512,  # how many channels to use in a spectrogram
    'num_freq_final': syll_size,  # how many channels to use in the resized spectrogram
    'sample_rate': 44100,  # what rate are your WAVs sampled at?
    'preemphasis': 0.97,
    'min_silence_for_spec': 0.5,  # minimum length of silence for a spectrogram to be considered a good spectrogram
    'frame_shift_ms': 2,  # step size for fft
    'frame_length_ms': 10,  # frame length for fft
    'min_level_db': -80,  # minimum threshold db for computing spe
    'spec_thresh_min': -20,  # (db)
    'spec_thresh_delta': 5,  # (db) what
    'ref_level_db': 70,  # reference db for computing spec
    'sample_rate': 44100,  # sample rate of your data
    'fmin': 1200,  # low frequency cutoff for mel filter
    'fmax': 10000,  # high frequency cutoff for mel filter

    # Vocal Envelope
    'smoothing': 'gaussian',  # 'none',
    'envelope_signal': "spectrogram",  # spectrogram or waveform, what to get the vocal envelope from
    'gauss_sigma_s': .001,
    'FOI_min': 4,  # minimum frequency of interest for vocal envelope (in terms of mel)
    'FOI_max': 24,  # maximum frequency of interest for vocal envelope (in terms of mel)

    # Silence Thresholding
    'silence_threshold': 0,  # normalized threshold for silence
    'min_len': 5.,  # minimum length for a vocalization (fft frames)
    'power_thresh': .3,  # Threshold for which a syllable is considered to be quiet weak and is probably noise

    # Syllabification
    'min_syll_len_s': 0.25,  # minimum length for a syllable
    'segmentation_rate': 0.0,
    # 0.125, # rate at which to dynamically raise the segmentation threshold (ensure short syllables)
    'threshold_max': 0.25,
    'min_num_sylls': 20,  # min number of syllables to be considered a bout
    'slow_threshold': 0.0,  # 0.02, # second slower threshold
    'max_size_syll': syll_size,  # the size of the syllable
    'resize_samp_fr': int(syll_size * 1.5),
    # (frames/s) the framerate of the syllable (in compressed spectrogram time components)

    # Sencond pass syllabification
    'second_pass_threshold_repeats': 50,  # the number of times to repeat the second pass threshold
    'ebr_min': 0.25,  # expected syllabic rate (/s) low
    'ebr_max': 1.25,  # expected syllabic rate (/s) high
    'max_thresh': 0.02,  # maximum pct of syllabic envelope to threshold at in second pass
    'thresh_delta': 0.005,  # delta change in threshold to match second pass syllabification
    'slow_threshold': 0.005,  # starting threshold for second pass syllabification

    'pad_length': syll_size,  # length to pad spectrograms to

    # spectrogram inversion
    'max_iters': 200,
    'griffin_lim_iters': 60,
    'power': 1.5,

    # Thresholding out noise
    'mel_noise_filt': 0.15,
    # thresholds out low power noise in the spectrum - higher numbers will diminish inversion quality
}
globals().update(hparams)
now_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # this is used to identify this training instance
# save the dictionary so that we can reload it for recovering waveforms
dict_dir = f'{os.path.expanduser("~")}/Data/bird-db/parameter_dictionaries'
if not os.path.isdir(dict_dir):
    os.mkdir(dict_dir)
dict_save = f'{dict_dir}/{now_string}_dict.pickle'
with open(dict_save, 'wb') as f:
    pickle.dump(hparams, f, protocol=pickle.HIGHEST_PROTOCOL)
print(dict_save)

### Mel Compression
_mel_basis = sg._build_mel_basis(hparams)  # build a basis function if you are using a mel spectrogram

# point toward your downloaded dataset
bird_species = glob(f'{os.path.expanduser("~")}/Data/bird-db/{dataset_name}*')
hdf5_save_loc = f'{os.path.expanduser("~")}/Data/bird-db/hd5f_save_loc'
if not os.path.exists(hdf5_save_loc):
    os.mkdir(hdf5_save_loc)

key_list = (
    'all_bird_wav_file',  # Wav file (bout_raw) that the syllable came from
    'all_bird_syll',  # spectrogram of syllable
    'all_bird_syll_start',  # time that this syllable occured
    'all_bird_t_rel_to_file',  # time relative to bout file that this
    'all_bird_syll_lengths',  # length of the syllable
    'all_bird_symbol',  # the symbolic representation of the syllable
)

dataset_sizes = {}

verbose = True
plot_syllables = True
max_syllable_length = 1.0
max_size = 32

### Parse textgrids
song_df = pd.DataFrame(
    columns=['bird', 'species', 'WavTime', 'WavLoc', 'WaveFileName', 'Position', 'Length', 'NumNote', 'NotePositions',
             'NoteLengths', 'NoteLabels'])
for species_folder in bird_species:
    species = species_folder.split('/')[-1]
    print(species)
    individuals = glob(species_folder + '/*')
    dataset_sizes[species] = []
    for individual_folder in tqdm(individuals, leave=False):
        individual = individual_folder.split('/')[-1]
        textgrids = glob(individual_folder + '/TextGrids/*.TextGrid')
        for textgrid_loc in tqdm(textgrids, leave=False):
            wav_time = datetime.strptime(textgrid_loc.split('/')[-1][:-9], "%Y-%m-%d_%H-%M-%S-%f")
            # load the textgrid
            try:
                tg = tgio.openTextgrid(textgrid_loc)
            except:
                print('TextGrid did not load')
                continue
            # extract song from tiers
            all_tiers = [tg.tierDict[tier].entryList for tier in tg.tierDict]
            main_tier = all_tiers[0]
            # create entry for symbolid df
            if len(np.array(main_tier).T) == 0:
                continue
            start_list, stop_list, label_list = np.array(main_tier).T
            # load the wav
            wav_loc = '/'.join(textgrid_loc.split('/')[:-2] + ['wavs'] + [textgrid_loc.split('/')[-1][:-9] + '.wav'])
            if not os.path.exists(wav_loc): continue
            with wave.open(wav_loc, "rb") as wave_file:
                rate = wave_file.getframerate()
            # create row
            song_df.loc[len(song_df)] = [individual, species, wav_time, wav_loc, wav_loc.split('/')[-1], None, None,
                                         len(main_tier),
                                         list((np.array([i.start for i in main_tier]) * rate).astype('int')),
                                         list((np.array([i.end - i.start for i in main_tier]) * rate).astype('int')),
                                         [i.label for i in main_tier],
                                         ]

### Get syllables from wavs
key_list = (
    'all_bird_wav_file',  # Wav file (bout_raw) that the syllable came from
    'all_bird_syll',  # spectrogram of syllable
    'all_bird_syll_start',  # time that this syllable occured
    'all_bird_t_rel_to_file',  # time relative to bout file that this
    'all_bird_syll_lengths',  # length of the syllable
    'all_bird_symbol',  # the symbolic representation of the syllable
)
parallel = True
verbosity = 0
n_jobs = 20
save = True

hparams['pct_fail'] = .8  # minimum percentage of spectral slice without power to be considered good
hparams['power_thresh'] = .25

for indv in tqdm(np.unique(song_df.bird)[::-1]):
    num_notes = np.sum(song_df[song_df.bird == indv].NumNote)
    if num_notes < 1000: continue
    print(indv, num_notes)
    if parallel:
        with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
            bird_data_packed = parallel(
                delayed(w2s.getSyllsFromWav)(row, _mel_basis, row.WavTime, hparams)
                for idx, row in tqdm(song_df[song_df.bird == indv].iterrows(),
                                     total=np.sum(song_df.bird == indv), leave=False))
    else:
        bird_data_packed = [w2s.getSyllsFromWav(row, _mel_basis, row.WavTime, hparams)
                            for idx, row in tqdm(song_df[song_df.bird == indv].iterrows(),
                                                 total=np.sum(song_df.bird == indv), leave=False)]
    bird_data_packed = [i for i in bird_data_packed if i is not None]
    bird_data_packed = [item for sublist in bird_data_packed for item in sublist]
    bird_data_packed = [i for i in bird_data_packed if i is not None]
    # initialize lists of bird information
    bird_data = {key: [] for key in key_list}
    for dtype, darray in zip(key_list, list(zip(*bird_data_packed))):
        for element in darray: bird_data[dtype].append(element)  # flatten and clear darray -> bird_data[dtype]
        bird_data[dtype] = np.array(bird_data[dtype])
    # reformat bird syllables
    print('len dataset: ', len(bird_data['all_bird_syll_lengths']))
    save_dir = f'{hdf5_save_loc}/{species}_wavs'
    save_name = indv.replace(" ", "_") + '_' + str(syll_size) + '.hdf5'
    save_loc = f'{save_dir}/{save_name}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if save:
        save_dataset(save_loc,
                     bird_data['all_bird_syll'],
                     bird_data['all_bird_syll_start'].astype('object'),
                     bird_data['all_bird_syll_lengths'],
                     bird_data['all_bird_wav_file'].astype('object'),
                     bird_data['all_bird_t_rel_to_file'],
                     bird_data['all_bird_symbol'],
                     indv
                     )
    if plot_syllables and len(bird_data['all_bird_syll']) > 0:  w2s.plt_all_syllables(bird_data['all_bird_syll'],
                                                                                      syll_size, max_rows=3,
                                                                                      max_sylls=100, width=900, zoom=1,
                                                                                      cmap=plt.cm.afmhot)
