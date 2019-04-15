import numpy as np
import pandas as pd
import os

import librosa
import pysndfile

from os import listdir, system
from os.path import isfile, join
import logging

def make_dir():
    # To train with silence clips, add folder _background_ folder with custom clips of background_noise
    folder_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                    'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    if not os.path.exists('UrbanSound8K/preprocessed'):
        os.makedirs('UrbanSound8K/preprocessed')
        for folder_name in folder_names:
            os.makedirs('UrbanSound8K/preprocessed/' + folder_name)

def get_audio_length(sound_clip, sample_rate):
    return len(sound_clip) / sample_rate

def get_dict():
    fn_class_dict = dict()
    raw_sound = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    for i in range(raw_sound.shape[0]):
        data_row = raw_sound.iloc[i:i+1]
        fn_class_dict[data_row['slice_file_name'].values[0]] = data_row['class'].values[0]
    return fn_class_dict

def split_files(dictionary):
    fold_list = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
    for i in range(10):
        mypath = 'UrbanSound8K/audio/'+ fold_list[i] + '/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for f in files:
            try:
                # Get file characteristics
                file_category = dictionary[f]
                file_path = mypath + f
                sound_clip, s = librosa.load(file_path, duration=1.0)
                # Write file to corresponding folder
                category_file_path = 'UrbanSound8K/preprocessed/' + file_category + "/" + f
                pysndfile.sndio.write(category_file_path, sound_clip, rate=16000, format='wav', enc='pcm16')
            except Exception as e:
                logging.exception("Cannot load file: " + f)
                continue
            fold = i+1
        

# Create the preprocessing directory and create corresponding label folders
make_dir()
fn_class_dict = get_dict()
# Split files according to labels
split_files(fn_class_dict)
