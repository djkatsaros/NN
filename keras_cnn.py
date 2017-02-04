'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
batch_size = 1#128
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 224,224
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#if K.image_dim_ordering() == 'th':
 #   X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  #  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
   # input_shape = (1, img_rows, img_cols)
#else:
 #   X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
  #  X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#NOTE: (batch, rows, cols) channel auto added
input_shape = (batch_size,img_rows, img_cols,3)

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=input_shape[1:]))#new
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides = (2,2)))
# model.add(Dropout(0.25))


model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))

model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides = (2,2)))

model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides = (2,2)))

model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))#new
# model.add(Convolution2D(nb_filters*8, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size,strides = (2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)

#train_datagen.fit()

generator = train_datagen.flow_from_directory(
        '/home/spiro/link_path/',
        target_size = (img_rows,img_cols),
        batch_size=10,
        class_mode='binary',
        classes = ['Benign','Malignant'])

#TODO:put real samples per epoch val in later
model.fit_generator(
        samples_per_epoch=20, 
        generator=generator,
        nb_epoch=nb_epoch,
        verbose=1)
#score = model.evaluate(X_test, Y_test, verbose=0)
