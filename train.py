'''
Command line script for training the VGG network
'''

import os
from libs import utils
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

import numpy as np
from argparse import ArgumentParser
from glob import glob

from keras.optimizers import Adam

from libs.data_generator import DataGenerator
from libs.speech_vgg import speechVGG

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_tqdm import TQDMCallback

def parse_args():
    parser = ArgumentParser(description='Training script for speechVGG')

    parser.add_argument(
        '-name', '--name',
        type=str, default='',
        help='Dataset name'
    )

    parser.add_argument(
        '-train', '--train',
        type=str,
        help='Folder with training images'
    )

    parser.add_argument(
        '-test', '--test',
        type=str,
        help='Folder with testing images'
    )

    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=32,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-weight_path', '--weight_path',
        type=str, default='./data/logs/',
        help='Where to output weights during training'
    )

    parser.add_argument(
        '-height', '--height',
        type=int, default=128,
        help='height of spectrogram'
    )

    parser.add_argument(
        '-width', '--width',
        type=int, default=128,
        help='width of spectrogram'
    )

    parser.add_argument(
        '-channels', '--channels',
        type=int, default=1,
        help='number of channels of spectrogram'
    )

    parser.add_argument(
        '-classes', '--classes',
        type=int, default=1000,
        help='number of classes (or words in our case)'
    )

    parser.add_argument(
        '-lr', '--lr',
        type=float, default=0.00005,
        help='learning rate'
    )

    parser.add_argument(
        '-log_path', '--log_path',
        type=str, default='./data/logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '-epochs', '--epochs',
        type=str, default='50',
        help='Number of training epochs'
    )

    parser.add_argument(
        '-augment', '--augment',
        type=str, default='yes',
        help='Augment training data? yes/no'
    )

    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # create dictionary of file names
    partition = {'train': glob(args.train + '/*.h5'),
                 'test': glob(args.test + '/*.h5')}

    if args.augment == 'yes':
        train_augment = True
        print('Applying data augmentation in training...')
    else:
        train_augment = False
        print('Training WITHOUT augmentation (?)')

    # Create training generator
    train_generator = DataGenerator(
        list_IDs=partition['train'],
        batch_size=args.batch_size,
        dim=(args.width, args.height, args.channels),
        classes=args.classes,
        data_augmentation=train_augment,
        shuffle=True
    )

    # Create test generator
    test_generator = DataGenerator(
        list_IDs=partition['test'],
        batch_size=args.batch_size,
        dim=(args.width, args.height, args.channels),
        classes=args.classes,
        data_augmentation=False,
        shuffle=True
    )

    # Build model
    model = speechVGG(
        include_top=True,
        input_shape=(args.width, args.height, args.channels),
        classes=args.classes,
        pooling=None
    )

    # Compile model
    model.compile(
        optimizer=Adam(args.lr),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    # Train
    model.fit_generator(
        train_generator,
        validation_data=test_generator,
        epochs=np.int(args.epochs),
        verbose=0,
        callbacks=[
            TensorBoard(
                log_dir=os.path.join(args.log_path, args.name),
                write_graph=False
            ),
            ModelCheckpoint(
                os.path.join(args.log_path, args.name, 'weights.{epoch:02d}-{loss:.2f}.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            TQDMCallback()
        ]
    )
