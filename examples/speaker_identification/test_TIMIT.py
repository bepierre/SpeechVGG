import numpy as np
from argparse import ArgumentParser
from glob import glob
import sys
sys.path.append('../..')
from libs.data_generator import *
from libs.speech_vgg import speechVGG
import os
os.chdir('../..')

def parse_args():
    parser = ArgumentParser(description='Training script for speechVGG')

    parser.add_argument(
        '--test',
        type=str,
        help='Folder with testing images'
    )

    parser.add_argument(
        '--weights',
        type=str,
        help=''
    )

    parser.add_argument(
        '--height',
        type=int, 
        default=128,
        help='height of spectrogram'
    )

    parser.add_argument(
        '--width',
        type=int, 
        default=128,
        help='width of spectrogram'
    )

    parser.add_argument(
        '--channels',
        type=int, 
        default=1,
        help='number of channels of spectrogram'
    )

    parser.add_argument(
        '--classes',
        type=int, 
        help='number of classes'
    )

    parser.add_argument(
        '--sliding_offset',
        type=int, 
        default=10,
        help='speed of the sliding window'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    args = parse_args()

    model = speechVGG(
            include_top=True,
            input_shape=(args.width, args.height, args.channels),
            classes=args.classes,
            pooling=None,
            weights=args.weights,
            transfer_learning=True
    )

    test_files = glob(args.test + '/*.h5')
    correct_preds = 0
    for i, tfile in enumerate(test_files):
        print('speaker {}'.format(i))
        h5f = h5py.File(tfile, 'r')

        spec_tmp = np.swapaxes(h5f['mag'][:], 0, 1)
        data_tmp = spec_tmp[..., np.newaxis]

        data_tmp[:,:,0] = log_standardize(data_tmp[:,:,0])
        data_tmp= np.delete(data_tmp, (128), axis=1)

        label_tmp = h5f['class'][0]
        h5f.close()
        
        windows = []
        for i in range(0, np.shape(data_tmp)[0], args.sliding_offset):
            if np.shape(data_tmp)[0] - i < 128:
                break
            window = data_tmp[i:i+128,:,:]
            windows.append(window)
        # last window
        window = data_tmp[-128:,:,:]
        windows.append(window)
        windows = np.array([x for x in windows])
        preds = model.predict(windows, steps=1)
        preds = np.mean(preds, axis=0)
        label_prediction = np.argmax(preds)
        
        print((label_tmp==label_prediction))
        correct_preds = correct_preds + (label_tmp==label_prediction)
    
    print(correct_preds)
    print('accuracy : {}'.format(correct_preds / args.classes))
