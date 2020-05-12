import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, exposure

from PIL import Image

from keras import Model, Input
from keras.applications import VGG16
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def make_lungs():
    path = 'All247images/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        img = 1.0 - np.fromfile(path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave('JSRT/' + filename[:-4] + '.png', img)
        print("Lung", i, filename)

def make_masks():
    path = 'All247images/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('scr/scratch/fold1/masks/left lung/' + filename[:-4] + '.gif')
        right = io.imread('scr/scratch/fold1/masks/right lung/' + filename[:-4] + '.gif')
        io.imsave('JSRT/masks/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
        print('Mask', i, filename)

def getOneHot(imgs):
    label_encoder = LabelEncoder()
    coded = label_encoder.fit_transform(imgs)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = coded.reshape(len(coded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

path = 'lung-segmentation-2d/'
lb = OneHotEncoder()
train_imgs = pickle.load(open(path + "train_images_512.pk",'rb'), encoding='bytes')
train_labels = pickle.load(open(path + "train_labels_512.pk",'rb'), encoding='bytes')
test_imgs = pickle.load(open(path + "test_images_512.pk",'rb'), encoding='bytes')

# Break the training and test sets into their component images
# Each image in the set is composed of 9 other images.
# Breaking them down allows us to train on more data.
train_xray = np.zeros((630, 171, 171, 3))
train_imgs = train_imgs.numpy().reshape(70, 512, 512, 3)
for j, image in enumerate(train_imgs):
    for i in range(3):
        for k in range(3):
            train_xray[(j * 9) + i * 3 + k] = image[i:i+171, k:k+171, :]

train_labels = np.repeat(train_labels.numpy(), 9)
train_labels = getOneHot(train_labels)

test_xray = np.zeros((180, 171, 171, 3))
test_imgs = test_imgs.numpy().reshape(20, 512, 512, 3)
for j, image in enumerate(test_imgs):
    for i in range(3):
        for k in range(3):
            test_xray[(j * 9) + i * 3 + k] = image[i:i + 171, k:k + 171, :]

def preprosses_images(im1, im2):
    path = 'Covid_Xrays/'

    # SAVE IMAGES FROM ARRAY FORM. Images are in sets of 9 duplicates.
    for i, image in enumerate(im1):
        if i % 9 != 0:
            continue
        io.imsave("{}image{}.png".format(path, i), image)

    for i, image in enumerate(im2):
        if i % 9 != 0:
            continue
        io.imsave("{}train_image{}.png".format(path, i), image)

    # RESIZE AND SAVE IN DESIRED FORMAT.
    im_shape = (256, 256)
    for i, filename in enumerate(os.listdir(path)):
        im = Image.open(path + filename).convert('L')
        im.resize(im_shape, Image.ANTIALIAS)
        im.save(path + filename[:-4] + '.jpeg', "JPEG")

make_lungs()
make_masks()

preprosses_images(train_xray, test_xray)


