import numpy as np
from numpy.lib.function_base import disp
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 8
IMAGE_SIZE = 224
UAVID_DIR =  r'D:/NCKH/UAVID_dataset/dataset_keras/'

test_path = []
val_path = []
val_ID_path = []

image_dir = UAVID_DIR +'test_resized'    #image_dir = D:/NCKH/UAVID_dataset/uavid_v1.5_official_release_image/seq1/Image
for step,file_name in enumerate(os.listdir(image_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
    image_path = image_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_v1.5_official_release_image/seq1/Image/000100.png
    test_path.append(image_path)


# validation image and mask path
image_dir = UAVID_DIR + 'validation_resized'
mask_dir = UAVID_DIR + 'validation_mask_resized' 
for step,file_name in enumerate(os.listdir(image_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
    image_path = image_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_v1.5_official_release_image/seq1/Image/000100.png
    val_path.append(image_path)

for step,file_name in enumerate(os.listdir(mask_dir)): # step = 1,2,3,4.. file_name = 000000.png, 000100.png
    mask_path = mask_dir + '/' + file_name # D:/NCKH/UAVID_dataset/uavid_v1.5_official_release_image/seq1/Labels/000100.png
    val_ID_path.append(mask_path)


colormap = np.array( [[0,0,0],
                     [128,0,0],
                     [128,64,128],
                     [0,128,0],
                     [128,128,0],
                     [64,0,128],
                     [192,0,192],
                     [64,64,0]],dtype=np.uint8)

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255
    return image

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor),axis=0)) # expand tensor axis.
    predictions = np.squeeze(predictions) #remove axis with length 1 our of array.
    predictions = np.argmax(predictions,axis=2) # return index with maximum value in array.
    return predictions

def decode_colormap(mask,colormap):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    print(np.shape(r))
    for i in range(0,NUM_CLASSES):
        idx = mask == i # take position that has corresponding label
        r[idx] = colormap[i,0]
        g[idx] = colormap[i,1]
        b[idx] = colormap[i,2]
    output = np.stack([r,g,b],axis=2)
    return output

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image,0.35,colored_mask,0.65,0)
    return overlay

def plot_samples_matplotlib(display_list,figsize=(5,3)):
    _, axes = plt.subplots(nrows=1,ncols=len(display_list),figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_prediction(image_list,colormap,model,test_set=1):
    for image_file in image_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor,model=model)
        prediction_colormap = decode_colormap(prediction_mask,colormap)
        overlay = get_overlay(image_tensor,prediction_colormap)
        if test_set == 1:
            plot_samples_matplotlib([image_tensor,image_tensor],figsize=(18,24))
        elif test_set == 0:
            plot_samples_matplotlib([image_tensor,overlay,prediction_mask],figsize=(18,24))


model = tf.keras.models.load_model('UAVid_model')
#plot_prediction(test_path[0],colormap,model=model)
plot_prediction(val_path[:2],colormap,model=model,test_set=0)

# ls = val_path[0]
# ls_mask = val_ID_path[0]
# img = read_image(ls)
# img_mask = read_image(ls_mask,mask=True)
# img_mask = np.array(img_mask).astype(np.uint8)

# overlay = get_overlay(img,img_mask)
# _, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,7))
# axes[0].imshow(tf.keras.preprocessing.image.array_to_img(img))
# axes[0].set_title("Original Image")
# axes[1].imshow(tf.keras.preprocessing.image.array_to_img(overlay))
# axes[1].set_title("Predicted Mask")
# axes[2].imshow(tf.keras.preprocessing.image.array_to_img(img_mask))
# axes[2].set_title("Labeled Image")
# plt.show()
