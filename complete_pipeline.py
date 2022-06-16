import streamlit as st
import json
import pandas as pd
import os

import random
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
K.set_image_data_format('channels_first')

import pandas as pd
import os
from tqdm import tqdm
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

import numpy as np
import pandas as pd

from patchify import patchify, unpatchify
from sklearn.metrics import jaccard_score

import keras.backend.tensorflow_backend as tfback

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.placeholder(dtype=float)

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def stretch_n(img, lower_percent=5, higher_percent=95):    
    '''
    adjusting the contrast of images and getting values in a range
    '''
    # https://www.kaggle.com/aamaia/rgb-using-m-bands-example
    out = np.zeros_like(img, dtype=np.float32)
    n = img.shape[2]
    for i in range(n):
        a = 0       # np.min(img)
        b = 1       # np.max(img)
        c = np.percentile(img[:, :, i], lower_percent)
        d = np.percentile(img[:, :, i], higher_percent)
        t = a + (img[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)

def load_image_patch(img):
	img = np.rollaxis(img, 0, 3)
	img = stretch_n(img)

	st.image(img[:,:,:3],caption = 'Input Image')
	st.markdown("""---""")

	patch_imgid = patchify(img,(160,160,8),step=160)
	patch_imgid = patch_imgid[:,:,0,:,:,:]

	patch_images = np.zeros((patch_imgid.shape[0]*patch_imgid.shape[1], 160, 160,8))
	c = 0
	for i in range(patch_imgid.shape[0]):
	    for j in range(patch_imgid.shape[1]):
	        #structuring for prediction
	        patch_images[c,:,:,:] = patch_imgid[i,j,:,:,:]
	        c+=1
	patch_images = 2 * np.transpose(patch_images, (0, 3, 1, 2)) - 1
	return patch_images
    

def pred(img):
	tr = [0.4, 0.3, 0.4, 0.4, 0.4, 0.6, 0.6, 0.5, 0.1, 0.2] #threshold
	patch_images = load_image_patch(img)

	st.header('Predictions')

	model = load_model('unet_model.h5',custom_objects={'jaccard_coef': jaccard_coef})

	pred_mask = model.predict(patch_images)
	pred_mask = np.transpose(pred_mask,(0, 2,3,1))

	for i in range(len(pred_mask)):
	    for j in range(10):
	        pred_mask[i,j,:,:] = pred_mask[i,j,:,:]>tr[j]
	unpatch = np.zeros((5,5,160,160,10))
	c = 0
	for i in range(5):
	    for j in range(5):
	        unpatch[i,j,:,:,:] = pred_mask[c,:,:,:]
	        c+=1
	unpatched_prediction = unpatchify(np.expand_dims(unpatch,axis=2),(800,800,10))

	return unpatched_prediction

def plot_mask(img):
	prediction = pred(img)
	classes = ['Buildings','Misc.','Road','Track','Trees',
               'Crops','Waterway', 'Standing water','Vehicle Large','Vehicle Small']
	for ele in range(10): #going through prediction of all classes
		fig,ax = plt.subplots(figsize=(25,10))
		st.image(prediction[:,:,ele],caption = 'predicted segmentation '+classes[ele])
		st.markdown("""---""")

		
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
	img = tiff.imread(uploaded_file)
	plot_mask(img)