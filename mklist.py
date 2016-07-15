from __future__ import print_function

import os
import cv2
import numpy
import random
import csv

data_path = 'raw/'

def wsave(file,imgs):
	with open(file,'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ')
		for line in imgs:
			writer.writerow(line)

def create_train_list():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2
    imgs = []
    imgs_ids = []
    subjects = []
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask1.tif'
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        #total = cv2.sumElems(img_mask)
	#if total[0] == 0.0 :
	#	continue
        subject = image_name.split('_')[0]
        img_id = image_name.split('.')[0]

        imgs_ids.append([img_id])

	image_name = image_mask_name
        imgs.append(('/'+image_name,'/'+image_mask_name))
	
        subjects.append(subject)
    unique_subjects= list(set(subjects))
    random.seed(1)
    random.shuffle(unique_subjects)
    validate_fraction = 0.1
    nt = int(len(unique_subjects)*(1-validate_fraction))
    print("validation subjects",unique_subjects[nt:])
    train_imgs = [imgs[i] for i,x in enumerate(subjects) if x in unique_subjects[:nt]]
    valid_imgs = [imgs[i] for i,x in enumerate(subjects) if x in unique_subjects[nt:]]        
    valid_ids = [imgs_ids[i] for i,x in enumerate(subjects) if x in unique_subjects[nt:]]	
    wsave('train.txt', train_imgs)
    wsave('val.txt', valid_imgs)
    wsave('val_id.txt',valid_ids)	

def create_test_list():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs=[]	
    imgs_id = []

    for image_name in images:
        img_id = image_name.split('.')[0]

        imgs.append(['/'+image_name])
        imgs_id.append([img_id])


    wsave('test.txt', imgs)
    wsave('test_id.txt', imgs_id)



if __name__ == '__main__':
    create_train_list()
    create_test_list()
