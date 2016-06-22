from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ProgbarLogger
from keras import backend as K

from data import load_train_data, load_test_data
from resnet import resnet
from elliptic_fourier_descriptors import reconstruct

#img_rows = 224
#img_cols = 224

img_rows = 64
img_cols = 64

org_rows = 420
org_cols = 580

smooth = 1.
def recon(efds):
        recon = reconstruct(efds,100)
        recon = recon.astype(np.int32)
        contours, hierarchy = cv2.findContours(bin_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        mask = np.zeros((org_row,org_cols), dtype=np.uint8)
        mask = cv2.fillPoly(mask,[recon])
        return mask

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, coeffs_train = load_train_data()
    masked = np.ma.masked_values(coeffs_train[:,0],0.0)
    
    imgs_train = imgs_train[~masked.mask,...]
    coeffs_train = coeffs_train[~masked.mask,...]

    imgs_train = preprocess(imgs_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    
    coeffs_train = coeffs_train[:,0:2]
    coeffs_train = coeffs_train.astype('float32')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = resnet(img_rows,img_cols)
    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')

    model_checkpoint = ModelCheckpoint('resnet.hdf5', monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #print (coeffs_train)
    prog = ProgbarLogger()
    model.fit(imgs_train, coeffs_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              callbacks=[prog,model_checkpoint],validation_split = 0.1)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting  on test data...')
    print('-'*30)
    coeffs_test = model.predict(imgs_test, verbose=1)
    np.save('coeffs_test.npy', coeffs_test)


if __name__ == '__main__':
    train_and_predict()
