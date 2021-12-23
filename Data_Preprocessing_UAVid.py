import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.nn_ops import convolution
from tensorflow.keras import regularizers
from collections import namedtuple

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024

BATCH_SIZE = 4
NUM_CLASSES = 8
UAVID_DIR = r'D:/NCKH/UAVID_dataset/dataset_keras/'
TRAIN_FOLDER_LIST = ['seq1','seq2','seq3','seq4','seq5','seq6','seq7','seq8',
    'seq8','seq9','seq10','seq11','seq12','seq13','seq14','seq15','seq31','seq32',
    'seq33','seq33','seq34','seq35']
VAL_FOLDER_LIST = ['seq16','seq17','seq18','seq19','seq20','seq36','seq37']
TEST_FOLDER_LIST = ['seq21','seq22','seq23','seq24','seq25','seq26','seq27',
    'seq28','seq29','seq30','seq38','seq39','seq40','seq41','seq42']


# get path of training image and corresponding mask
# train_path = []
# train_ID_path = []
# val_path = []
# val_ID_path = []
# test_path = []

'''Quantize the Palette '''

def QuantizeToGivenPalette(im, palette):
    """Quantize image to a given palette.

    The input image is expected to be a Numpy array.
    The palette is expected to be a list of R,G,B values."""

    # Calculate the distance to each palette entry from each pixel
    distance = np.linalg.norm(im[:,:,None] - palette[None,None,:], axis=3)

    # Now choose whichever one of the palette colours is nearest for each pixel
    palettised = np.argmin(distance, axis=2).astype(np.uint8)

    return palettised

# Open input image and palettise to "inPalette" so each pixel is replaced by palette index
# ... so all black pixels become 0, all red pixels become 1, all green pixels become 2...
#im=cv2.imread(path,cv2.IMREAD_COLOR)

'''Color of palette'''
inPalette = np.array([
   [0,0,0],             # background clutter
   [128,0,0],           # building 
   [128,64,128],        # road
   [0,128,0],           # tree
   [128,128,0],         # low vegetation
   [64,0,128],          # moving car
   [192,0,192],         # static car
   [64,64,0]] ,         # human
   dtype=np.uint8)

#r = QuantizeToGivenPalette(im,inPalette)

# Now make LUT (Look Up Table) with the 5 new colours
LUT = np.zeros((8,3),dtype=np.uint8)
LUT[0]=[0,0,0]        # label 0
LUT[1]=[1,1,1]        # label 1
LUT[2]=[2,2,2]        # label 2
LUT[3]=[3,3,3]        # label 3
LUT[4]=[4,4,4]        # label 4
LUT[5]=[5,5,5]        # label 5
LUT[6]=[6,6,6]        # label 6
LUT[7]=[7,7,7]        # label 7
# Look up each pixel in the LUT
# result = LUT[r]

# ---------------------------------------------------------------- #

''' Extract path of training image & mask.
1) Read and resize image & mask to  (512x1024) resolution
2) Save new resized image to new directory
3) Map id to mask image
4) Save new resized mask to new directory
'''

for dir_step,dir in enumerate(TRAIN_FOLDER_LIST): # dirstep = 1,2,3,4.. dir = 'seq1','seq2'
    image_dir = UAVID_DIR +'uavid_train/' +  dir + '/' + 'Images' #image_dir = D:/NCKH/UAVID_dataset/uavid_train/seq1/Image
    mask_dir = UAVID_DIR + 'uavid_train/' + dir + '/' + 'Labels'
    for step,file_name in enumerate(os.listdir(image_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png    
        image_path = image_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_train/seq1/Image/000100.png
        read_image = cv2.imread(image_path,-1) # read image out
        take_name = file_name.split('.')[0] # take the name of image
        resized_image_dir = UAVID_DIR + 'train_resized/' + dir + '_' + take_name + '.png'
        resized_image = cv2.resize(read_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_NEAREST) # nearest neighbor interpolation
        cv2.imwrite(resized_image_dir,resized_image ) # save resized image to 'image_resized' folder

    for step,file_name in enumerate(os.listdir(mask_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
        mask_path = mask_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_train/seq1/Label/000100.png
        read_mask = cv2.imread(mask_path,-1) #read mask
        take_name = file_name.split('.')[0] # take the name of mask
        resized_mask_dir = UAVID_DIR + 'train_mask_resized/' + dir + '_' + take_name + '.png'
        resized_ID_mask_dir = UAVID_DIR + 'train_ID_mask_resized/' + dir + '_' + take_name + '.png'
        resized_mask = cv2.resize(read_mask,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_NEAREST)
        quantized_mask = QuantizeToGivenPalette(read_mask,inPalette)
        conversed_ID_mask = LUT[quantized_mask]
        cv2.imwrite(resized_mask_dir,resized_mask)
        cv2.imwrite(resized_ID_mask_dir,conversed_ID_mask)

''' Extract path of validation image & mask.
1) Read and resize image & mask to  (512x1024) resolution
2) Save new resized image to new directory
3) Map id to mask image
4) Save new resized mask to new directory
'''

for dir_step,dir in enumerate(VAL_FOLDER_LIST): # dirstep = 1,2,3,4.. dir = 'seq1','seq2'
    image_dir = UAVID_DIR + 'uavid_val/' + dir + '/' + 'Images' #image_dir = D:/NCKH/UAVID_dataset/uav_test/seq16/Image
    mask_dir = UAVID_DIR + 'uavid_val/' + dir + '/' + 'Labels'
    for step,file_name in enumerate(os.listdir(image_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
        image_path = image_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uav_test/seq16/Image/000100.png
        read_image = cv2.imread(image_path,-1)
        take_name = file_name.split('.')[0] # take the name of image
        resized_image_dir = UAVID_DIR + 'validation_resized/' + dir + '_' + take_name + '.png'
        resized_image = cv2.resize(read_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_NEAREST) # nearest neighbor interpolation
        cv2.imwrite(resized_image_dir,resized_image ) # save resized image to 'image_resized' folder

    for step,file_name in enumerate(os.listdir(mask_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
        mask_path = mask_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uav_test/seq16/Label/000100.png
        read_mask = cv2.imread(mask_path,-1) 
        take_name = file_name.split('.')[0] # take the name of mask
        resized_mask_dir = UAVID_DIR + 'validation_mask_resized/' + dir + '_' + take_name + '.png'
        resized_ID_mask_dir = UAVID_DIR + 'validation_ID_mask_resized/' + dir + '_' + take_name + '.png'
        resized_mask = cv2.resize(read_mask,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_NEAREST)
        quantized_mask = QuantizeToGivenPalette(read_mask,inPalette)
        conversed_ID_mask = LUT[quantized_mask]
        cv2.imwrite(resized_mask_dir,resized_mask)
        cv2.imwrite(resized_ID_mask_dir,conversed_ID_mask)



''' Extract path of test image & mask.
1) Read and resize image to  (512x1024) resolution
2) Save new resized image to new directory
'''
for dir_step,dir in enumerate(TEST_FOLDER_LIST): # dirstep = 1,2,3,4.. dir = 'seq1','seq2'
    image_dir = UAVID_DIR +'uavid_test/' +  dir + '/' + 'Images' #image_dir = D:/NCKH/UAVID_dataset/uavid_test/seq21/Image
    for step,file_name in enumerate(os.listdir(image_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png    
        image_path = image_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_test/seq21/Image/000100.png
        read_image = cv2.imread(image_path,-1) # read image out
        take_name = file_name.split('.')[0] # take the name of image
        resized_image_dir = UAVID_DIR + 'test_resized/' + dir + '_' + take_name + '.png'
        resized_image = cv2.resize(read_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_NEAREST) # nearest neighbor interpolation
        cv2.imwrite(resized_image_dir,resized_image ) # save resized image to 'image_resized' folder


