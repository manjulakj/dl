import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import argparse
import zipfile
import math

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import random as rn
import tensorflow as tf
import keras as k
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import *
from functools import partial
import tensorflow.keras.backend as K
from itertools import product
from keras import backend as kb
import horovod.tensorflow.keras as hvd
from azureml.core import Run
from tensorflow.keras.metrics import AUC


#################################################################
## python effusion_detector.py --data-folder '/tmp/data/data.zip'
#################################################################


# Horovod: initialize Horovod.
hvd.init()


# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

rn.seed(30)
np.random.seed(30)
tf.compat.v1.random.set_random_seed(30)
warnings.simplefilter('ignore')

# There are two classes of images that we will deal with
disease_cls = ['effusion', 'nofinding']

# preprocess images for training and validation, for training transformations are applied randomly.
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0,
    height_shift_range=0,
    vertical_flip=False,)

# preprocess images for training and validation, for training transformations are applied randomly.
def preprocess_img(img, mode):
    normalizedImg = np.zeros((256, 256))
    resized_img = cv2.resize(img, (256,256))
    normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)[:,:,np.newaxis]
    if mode == 'train':
        if np.random.randn() > 0:
            kernel = np.ones((5,5),np.uint8)
            opened = cv2.morphologyEx(normalized_img, cv2.MORPH_OPEN, kernel)[:,:,np.newaxis]
            normalized_img = datagen.random_transform(opened)
    return normalized_img

# creates and returns a cnn model
def prepare_model(img_rows, img_cols, img_channels, nb_classes):
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_rows,img_cols,img_channels)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu',))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu',))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(512))
    cnn.add(Dense(512))
    cnn.add(Dense(nb_classes,activation='sigmoid'))
    cnn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    print(cnn.summary())
    return cnn;

# this class ensures there is continuous stream of data to the model for training
class AugmentedDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, mode='train', ablation=None, disease_cls = ['nofinding', 'effusion'], 
                 batch_size=15, dim=(256, 256), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = {}
        self.list_IDs = []
        self.mode = mode
        
        for i, cls in enumerate(disease_cls):
            print('navigating data path:',os.path.join(DATASET_PATH, cls, '*'))
            paths = glob.glob(os.path.join(DATASET_PATH, cls, '*'))
            brk_point = int(len(paths)*0.8)
            if self.mode == 'train':
                paths = paths[:brk_point]
            else:
                paths = paths[brk_point:]
            if ablation is not None:
                paths = paths[:int(len(paths)*ablation/100)]
            self.list_IDs += paths
            self.labels.update({p:i for p in paths})
        
            
        self.n_channels = n_channels
        self.n_classes = len(disease_cls)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        delete_rows = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(ID, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
            if img.shape[:3] == (1024, 1024,1):
                img = preprocess_img(img, self.mode)
                X[i,] = img
                y[i] = self.labels[ID]
            else:
                delete_rows.append(i)
                continue
        X = np.delete(X, delete_rows, axis=0)
        y = np.delete(y, delete_rows, axis=0)
        return X, y

# calculates "validation auc" for every epoch
class roc_callback(Callback):
    
    def on_train_begin(self, logs={}):
        logs['val_auc'] = 0

    def on_epoch_end(self, epoch, logs={}):
        y_p = []
        y_v = []
        for i in range(len(validation_generator)):
            x_val, y_val = validation_generator[i]
            y_pred = self.model.predict(x_val)
            y_p.append(y_pred)
            y_v.append(y_val)
        y_p = np.concatenate(y_p)
        y_v = np.concatenate(y_v)
        roc_auc = roc_auc_score(y_v, y_p)
        print ('\nVal AUC for epoch{}: {}'.format(epoch, roc_auc))
        logs['val_auc'] = roc_auc
        run.log('Loss', logs['val_loss'])
        run.log('Accuracy', logs['val_accuracy'])
        run.log('AUC', logs['val_auc'])

# applys decaying learning rate for each epoch
class DecayLR(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=0.01, decay_epoch=1):
        super(DecayLR, self).__init__()
        self.base_lr = base_lr
        self.decay_epoch = decay_epoch 
        self.lr_history = []
        
    def on_train_begin(self, logs={}):
        kb.set_value(self.model.optimizer.lr, self.base_lr)

    def on_epoch_end(self, epoch, logs={}):
        new_lr = self.base_lr * (0.5 ** (epoch // self.decay_epoch))
        self.lr_history.append(kb.get_value(self.model.optimizer.lr))
        kb.set_value(self.model.optimizer.lr, new_lr)

# extracting data
def prep_data(filename):
    directory_to_extract_to = '/var/tmp/effusion/'
    os.makedirs(directory_to_extract_to, exist_ok=True)
    with zipfile.ZipFile(filename) as f:
       print('Extracting files to...', directory_to_extract_to)
       f.extractall(directory_to_extract_to)
    return directory_to_extract_to + 'data/'

print("TensorFlow version:", tf.__version__)


# loading arguments from job definition
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=32, help='mini batch size for training')
parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='Number of Epochs')
args = parser.parse_args()
DATASET_PATH = prep_data(args.data_folder)

# few constants
n_inputs = 256 * 256

# Horovod: adjust number of epochs based on number of GPUs.
n_epochs = int(math.ceil(args.epochs / hvd.size())) 
batch_size = args.batch_size
print('Data Folder', DATASET_PATH)
print('Batch Size', batch_size)
print('Epochs', n_epochs)

# Build neural network model.
neural_net = prepare_model(256, 256, 1, 1)

# start an Azure ML run
run = Run.get_context()

# start timer
start_time = time.perf_counter()

os.makedirs('./outputs/model', exist_ok=True)
filepath = './outputs/model/best_model.h5'

# initiating call backs
auc_logger = roc_callback()
decay = DecayLR()
checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
# Horovod: broadcast initial variable states from rank 0 to all other processes.
# This is necessary to ensure consistent initialization of all workers when
# training is started with random weights or restored from a checkpoint.
broadcastGV = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
ES = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, verbose=1, mode='max')

training_generator = AugmentedDataGenerator('train', ablation=None)
validation_generator = AugmentedDataGenerator('val', ablation=None)

callbacks_list = [auc_logger, decay, broadcastGV, ES]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks_list.append(checkpoint)

history = neural_net.fit(training_generator, epochs=n_epochs, validation_data=validation_generator, verbose=2, callbacks=callbacks_list)

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))