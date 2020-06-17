# speaker-identification

#### Data :

You should create a folder 'TIMIT' with the following folders :

    TIMIT
    |_ TEST
            |____ DR1
            |____ DR2
            |____ DR3
            |____ DR4
            |____ DR5
            |____ DR6
            |____ DR7
            |____ DR8
    |_ TRAIN
            |____ DR1
            |____ DR2
            |____ DR3
            |____ DR4
            |____ DR5
            |____ DR6
            |____ DR7
            |____ DR8
            

### Generate dataset

First, preprocess the data with preprocess_TIMIT in the examples folder (task is either speaker or dialect):

```bash
python preprocess_TIMIT.py --data ~/TIMIT --dest_path ~/TIMIT_speakers --task speaker
```

Now move back to the root folder.

Obtain the mean and standard deviation of the desired dataset (for normalization):

```bash
python compute_dataset_props.py --data ~/TIMIT_speakers/TRAIN/ --output_folder data
```

Parameters will be saved as `dataset_props_log.h5` file. 

### Train

Now you can train the model using the training script (speakers in this case) using a pretrained network on another task (LibriSpeech words):

```bash
python train.py --train ~/TIMIT_speakers/TRAIN --test ~/TIMIT_speakers/TEST --augment yes --classes 630 --lr 0.00005 --weights ~/speechVGG_models/460hours_3000words_weights19-0.28.h5  --epochs 1000 --model speechVGG --transfer_learning yes 
```

notice how transfer learning is turned on: this means that it initializes all layers to those of libri speechVGG except the last linear layers (whose shapes depend on the number of classes).

### Test

Use the test script in the examples folder:

```bash
python test_TIMIT.py --test ~/TIMIT_speakers/TEST --weights data/weights/*someweight*  --classes 630 --sliding_offset 10
```

which, with a sliding window, averages softmax probs for a whole audio clip (model was trained only on 1 second segments)
