'''
Command line script for computing mean & variance based on the training data to
perform normalization later on in the process.
'''
import numpy as np
import glob
import h5py
import os

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Compute mean and variance of data.')

    parser.add_argument(
        '-data', '--data',
        type=str,
        help='Folder with spectrograms'
    )

    # TO DO: mkdir -p?
    parser.add_argument(
        '-output_folder', '--output_folder',
        type=str,
        default='./',
        help='Output folder to save the params file'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # create dictionary of file names
    filenames = glob.glob(args.data + '/*.h5')

    mean = 0
    sumsqmean = 0
    var = 0

    for filename in filenames:
        h5f = h5py.File(filename, 'r')
        spec = np.swapaxes(h5f['mag'][:], 0, 1)
        spec = np.log(spec)
        # same transform is used in data_generator
        spec[spec < -80] = -80
        mean_tmp = np.mean(spec, axis=0)
        mean = mean + mean_tmp
        sumsqmean = sumsqmean + mean_tmp * mean_tmp

    n = len(filenames)
    mean = mean / n
    var = (sumsqmean - (mean * mean) / n) / (n - 1)

    h5f = h5py.File(os.path.join(args.output_folder, 'dataset_props_log.h5'), 'w')
    h5f.create_dataset('mean', data=mean)
    h5f.create_dataset('var', data=var)
    h5f.close()
