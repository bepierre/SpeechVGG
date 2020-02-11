'''
Updated version of the module containing data generators for the VGG network.
'''

import numpy as np
import keras
import h5py

from numpy.random import randint

def one_hot_index(index, classes):
    'One-hot encoding'
    return np.array([int(i==index) for i in range(classes)])

def log_standardize(spec, path_norm='dataset_props_log.h5'):
    'Apply log operation to the spectrogram and normalize it based on the training data.'
    spec = np.log(spec)
    spec[spec < -80] = -80

    h5f = h5py.File(path_norm, 'r')
    mean = h5f['mean'][:]
    var = h5f['var'][:]
    h5f.close()

    mean = mean[..., np.newaxis].T
    var = var[..., np.newaxis].T

    spec = (spec - mean) / var

    return spec

def get_augment_mask(spec, maxtimenum=2, maxtimesize=100, maxfreqnum=2, maxfreqsize=100):
    'Create spec_augment mask'
    w,h,c = spec.shape
    aug_mask = np.zeros((w,h,c), dtype=int)

    for i in range(maxtimenum):
        hole_size = randint(0, min(maxtimesize, int(w/4)))
        start = randint(0, w - hole_size)
        aug_mask[start:start + hole_size,:,:] = 1

    for i in range(maxfreqnum):
        hole_size = randint(1, min(maxfreqsize, int(h/4)))
        start = randint(0, h - hole_size)
        aug_mask[:,start:start + hole_size,:] = 1

    return aug_mask

def apply_augment_mask(spec, mask, mode='zero'):
    'Apply spec_augment mask to the spectrogram'
    assert spec.shape == mask.shape
    if mode == 'mean':
        pad_val = np.mean(spec)*np.ones(mask.shape)
    elif mode == 'random_phase':
        pad_val = (np.random.rand(*mask.shape)*2*np.pi) - np.pi
    else:
        pad_val = np.zeros(mask.shape)

    spec[mask==1] = pad_val[mask==1]

    return spec

def pad_spec(spec, width=128):
    '''Pad word spectrogram to the right size (default: 128)'''
    #spec = np.delete(spec, (128), axis=1)
    to_add = width - np.size(spec,0)
    if to_add > 0:
        left = np.random.randint(0, to_add)
        right = to_add - left
        spec = np.pad(spec, pad_width=((left, right), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif to_add < 0:
        left = np.random.randint(0, -to_add)
        right = - to_add - left
        spec = spec[left:np.size(spec,0)-right,:,:]
    else:
        pass
    return spec

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=32, dim=(128,128,1), classes=1000, data_augmentation=True, shuffle=True, phase_scaling=1/60., phase_init='random'):
        'Initialization'
        self.dim = dim  # If phase is to be used last dimension > 1
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.classes = classes
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.on_epoch_end()
        self.phase_scaling = phase_scaling  # If phase is used
        self.phase_init = phase_init  # If phase is used choose gap-filling mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        specs, labels = self.__data_generation(list_IDs_temp)

        return specs, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        data = np.empty((self.batch_size, *self.dim))
        labels = np.empty((self.batch_size, self.classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                h5f = h5py.File(ID, 'r')

                spec_tmp = np.swapaxes(h5f['mag'][:], 0, 1)
                spec_tmp = spec_tmp[..., np.newaxis]
                if self.data_augmentation:
                    aug_mask = get_augment_mask(spec_tmp)
                    spec_tmp = apply_augment_mask(spec_tmp, aug_mask, mode='mean')

                if self.dim[-1] > 1:
                    phase_tmp = np.swapaxes(h5f['phase'][:], 0, 1)
                    phase_tmp = phase_tmp[..., np.newaxis]
                    if self.data_augmentation:
                        if self.phase_init == 'random':
                            phase_tmp = apply_augment_mask(phase_tmp, aug_mask, mode='random_phase')
                        else:
                            phase_tmp = apply_augment_mask(phase_tmp, aug_mask, mode='zero')
                    phase_tmp = phase_tmp*self.phase_scaling
                    data_tmp = np.dstack((spec_tmp, phase_tmp))
                else:
                    data_tmp = spec_tmp[:]

                data_tmp = pad_spec(data_tmp)
                data_tmp[:,:,0] = log_standardize(data_tmp[:,:,0])
                data_tmp= np.delete(data_tmp, (128), axis=1)
                data[i,] = data_tmp

                label_tmp = h5f['word_idx'][0]
                label_tmp = one_hot_index(label_tmp, self.classes)
                labels[i,] = label_tmp
                h5f.close()

            except ValueError:
                print('Error at: {}'.format(ID))
                print('Wrong shape? {}'.format(spec_tmp.shape))
                print('Wrong shape? {}'.format(data_tmp.shape))

        return data, labels
