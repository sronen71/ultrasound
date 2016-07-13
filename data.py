from __future__ import print_function

import os
import numpy as np

import cv2
from  elliptic_fourier_descriptors import elliptic_fourier_descriptors as efd
import random

data_path = 'raw/'

image_rows = 420
image_cols = 580

degree = 2

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    coeffs = np.ndarray((total, 4*degree-3),dtype = np.float)
    subjects = np.ndarray(total,dtype = np.uint8)
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
	image_mask_write = image_name.split('.')[0] + '_mask1.tif'
        subject = image_name.split('_')[0]
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_write), cv2.IMREAD_GRAYSCALE)
        print(np.max(img_mask)) 
	
	img_mask[img_mask==255] = 1
        print(img_mask[img_mask>0]) 
	cv2.imwrite(os.path.join(train_data_path,image_mask_write),img_mask)	
        efds,K,T = efd(img_mask,degree)
        if efds.size:
                efds1 = efds[0]
                efds1 = efds1.reshape((efds1.size))
                coeff = np.delete(efds1,[1,3,4])
        else :
                coeff = np.zeros(4*degree-3)
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
        coeffs[i] = coeff
        subjects[i] = subject
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    unique_subjects= np.unique(subjects).tolist()
    random.seed(1)
    random.shuffle(unique_subjects)
    validate_fraction = 0.1
    nt = int(len(unique_subjects)*(1-validate_fraction))
    print("validation subjects",unique_subjects[nt:])
    train_index = [i for i,x in enumerate(subjects.tolist()) if x in unique_subjects[:nt]]
    valid_index = [i for i,x in enumerate(subjects.tolist()) if x in unique_subjects[nt:]]        
    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    np.save('imgs_train.npy', imgs[train_index,...])
    np.save('imgs_mask_train.npy', imgs_mask[train_index,...])
    print(coeffs.shape,train_index.shape,coeffs[train_index,...].shape) 	
    np.save('coeffs_train.npy',coeffs[train_index,...])
    np.save('imgs_valid.npy', imgs[valid_index,...])
    np.save('imgs_mask_valid.npy', imgs_mask[valid_index,...])
    np.save('coeffs_valid.npy',coeffs[valid_index,...])


    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    coeffs_train = np.load('coeffs_train.npy')
    return imgs_train, coeffs_train


def load_valid_data():
    imgs_valid = np.load('imgs_valid.npy')
    coeffs_valid = np.load('coeffs_valid.npy')
    return imgs_valid, coeffs_valid


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
