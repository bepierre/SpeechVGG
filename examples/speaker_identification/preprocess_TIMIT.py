import os

import numpy as np

from argparse import ArgumentParser

import soundfile as sf
from scipy.signal import stft

import h5py

from glob import glob

import random

def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles for TIMIT')

    parser.add_argument(
        '--data',
        type=str,
        help='dataset to load'
    )

    parser.add_argument(
        '--dest_path',
        type=str,
        help='destination of processed data'
    )

    parser.add_argument(
        '--task',
        type=str,
        choices=['speaker', 'dialect'],
        help='task to preprocess to'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    if args.task=='speaker':
        # create dest folders
        os.mkdir('{}/TEST'.format(args.dest_path))
        os.mkdir('{}/TRAIN'.format(args.dest_path))

        # get all wav files
        audio_files = glob('{}/*/*/*/*.WAV'.format(args.data))
        random.shuffle(audio_files)

        
        speakers = [audio_file.split('/')[-2] for audio_file in audio_files]
        
        speakers_dict = list(dict.fromkeys(speakers))
        classes = [speakers_dict.index(speaker) for speaker in speakers]
        mylist = list(dict.fromkeys(classes))

        # save a file for each data point
        i=0
        to_test = np.ones((len(mylist), 1), dtype=bool)
        for audio_file, audio_class in zip(audio_files, classes):

            audio = sf.read(audio_file)
            audio = audio[0]
            f, t, seg_stft = stft(audio,
                                window='hamming',
                                nperseg=256,
                                noverlap=128)
            mag_spec = np.abs(seg_stft)
            
            if to_test[audio_class]:
                file_dest = '{}/TEST/{}_{}.h5'.format(args.dest_path, audio_file.split('/')[-1][:-4], i)
                to_test[audio_class] = False
            else:
                file_dest = '{}/TRAIN/{}_{}.h5'.format(args.dest_path, audio_file.split('/')[-1][:-4], i)
            h5f = h5py.File(file_dest, 'w')
            h5f.create_dataset('class', data=[audio_class])
            h5f.create_dataset('mag', data=mag_spec)
            h5f.close()
            i = i + 1
    
    elif args.task=='dialect':
        # get all wav files
        audio_files = glob('{}/*/*/*.WAV'.format(args.data))
        
        speakers = [audio_file.split('/')[-3] for audio_file in audio_files]
        
        speakers_dict = list(dict.fromkeys(speakers))
        classes = [speakers_dict.index(speaker) for speaker in speakers]

        # save a file for each data point
        i=0
        for audio_file, audio_class in zip(audio_files, classes):

            audio = sf.read(audio_file)
            audio = audio[0]
            f, t, seg_stft = stft(audio,
                                window='hamming',
                                nperseg=256,
                                noverlap=128)
            mag_spec = np.abs(seg_stft)
            
            file_dest = '{}/{}_{}.h5'.format(args.dest_path, audio_file.split('/')[-1][:-4], i)
            file_dest = '{}/{}_{}.h5'.format(args.dest_path, audio_file.split('/')[-1][:-4], i)
            h5f = h5py.File(file_dest, 'w')
            h5f.create_dataset('class', data=[audio_class])
            h5f.create_dataset('mag', data=mag_spec)
            h5f.close()
            i = i + 1
