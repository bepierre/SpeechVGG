'''
Command-line script for preprocessing LibriSpeech data for training the VGG
deep feature extractor. It involves, loading the soundfiles, parsing them to
obtain single words extracting features (STFT) and saving them.
'''

import os

import numpy as np

from argparse import ArgumentParser

import soundfile as sf
from scipy.signal import stft

import h5py

def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles')

    parser.add_argument(
        '-data', '--data',
        type=str, default='./LibriSpeech',
        help='dataset to load'
    )

    parser.add_argument(
        '-dest_path', '--dest_path',
        type=str, default='./LibriSpeech_words',
        help='destination of processed data'
    )

    parser.add_argument(
        '-classes', '--classes',
        type=int, default=1000,
        help='number of classes (in our case words)'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # splits
    split_dir = args.data+'/split/'
    splits = [ name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name)) ]
    dict_file = args.data + '/word_labels/{}-mostfreq'.format(args.classes)
    dictionary = open(dict_file, "r").read().split('\n')
    if dictionary[-1] == '': del dictionary[-1]

    for split in splits:

        # Create output directories if don't exist
        os.makedirs(args.dest_path + '/' + split, exist_ok=True)

        print('Pre-processing: {}'.format(split))

        # Get file names
        word_file = args.data + '/word_labels/' + split + '-selected-' + str(args.classes) + '.txt'

        current_file_name = ''
        audio = 0

        with open(word_file) as wf:

            segment_num = 0

            for line in wf.readlines():

                # remove endline if present
                line = line[:line.find('\n')]
                segment_name, _, time_beg, time_len, word, _ = line.split(' ')

                file_name = args.data + '/split/' + split + '/' + segment_name.replace('-', '/')[:segment_name.rfind('-')+1] + segment_name + '.flac'
                if file_name != current_file_name:
                    audio = sf.read(file_name)
                    audio = audio[0]
                    current_file_name = file_name
                    segment_num = 0

                start = int(float(time_beg) * 16000)
                end = int((float(time_beg) + float(time_len)) * 16000)

                f, t, seg_stft = stft(audio[start:end],
                                      window='hamming',
                                      nperseg=256,
                                      noverlap=128)
                

                h5f = h5py.File(args.dest_path + '/' + split + '/' + segment_name + '_' + str(segment_num) + '.h5', 'w')
                h5f.create_dataset('word_idx', data=[dictionary.index(word.lower())])
                h5f.create_dataset('mag', data=np.abs(seg_stft))
                h5f.close()

                segment_num = segment_num + 1
