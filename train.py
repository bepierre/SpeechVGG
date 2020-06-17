'''
Command line script for training the VGG network
'''

import os
from libs import utils
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

import numpy as np
from argparse import ArgumentParser
from glob import glob

from keras.optimizers import Adam

from libs.data_generator import DataGenerator
from libs.speech_vgg import speechVGG

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras_tqdm import TQDMCallback
from tensorflow.python.client import device_lib

# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# tf.test.is_gpu_available()

def parse_args():
    parser = ArgumentParser(description='Training script for speechVGG')

    parser.add_argument(
        '--name',
        type=str, 
        default='',
        help='Dataset name'
    )

    parser.add_argument(
        '--train',
        type=str,
        help='Folder with training images'
    )

    parser.add_argument(
        '--test',
        type=str,
        help='Folder with testing images'
    )

    parser.add_argument(
        '--batch_size',
        type=int, 
        default=32,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '--weight_path',
        type=str, 
        default='./data/weights/',
        help='Where to output weights during training'
    )

    parser.add_argument(
        '--height',
        type=int, 
        default=128,
        help='height of spectrogram'
    )

    parser.add_argument(
        '--width',
        type=int, default=128,
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
        default=8,
        help='number of classes (or words in our case)'
    )

    parser.add_argument(
        '--lr',
        type=float, 
        default=0.00005,
        help='learning rate'
    )

    parser.add_argument(
        '--log_path',
        type=str, 
        default='./data/logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '--epochs',
        type=str, 
        default='50',
        help='Number of training epochs'
    )

    parser.add_argument(
        '--augment',
        type=str, 
        default='yes',
        help='Augment training data? yes/no'
    )

    parser.add_argument(
        '--weights',
        type=str,
        help='weights from where to start training'
    )

    parser.add_argument(
        '--transfer_learning',
        type=str, 
        default='no',
        help='Augment training data? yes/no'
    )

    parser.add_argument(
        '--model',
        default='speechVGG',
        type=str
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

    if args.transfer_learning == 'yes':
        transfer_learning = True
    else:
        transfer_learning = False

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
        shuffle=False
    )

    # Build model
    # model = sVGG_classifier(sVGG_weights=None,
    #         classes=args.classes,
    #         pooling=None,
    #         slide_offset=20)
    if args.model=='speechVGG':
        model = speechVGG(
            include_top=True,
            input_shape=(args.width, args.height, args.channels),
            classes=args.classes,
            pooling=None,
            weights=args.weights,
            transfer_learning=transfer_learning
        )
    elif args.model=='svgg_extractor':
        model = sVGG_extractor(
            input_shape=(args.width, args.height, args.channels),
            classes=args.classes,
            svgg_weights=args.weights
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
                os.path.join(args.weight_path, args.name, 'weights.{epoch:02d}-{loss:.2f}.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            TQDMCallback()
        ]
    )
