import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.image import imread
from pprint import pprint

# suppress Log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force CPU to calculate by hiding GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



image = imread('./100_100.png')
images = image[np.newaxis, :, :, :]
# must contain depth axis
reds = images[:,:,:,0:1]
# shape [most outer loop, second loop, third loop, ... , most inner loop]
# images.shape [1, 100, 100, 3] = [batch, row, col, depth(RGBA)]
# print(images.shape)  # ---> [1, 100, 100, 1]

# image = image.
# print(tf.__version__)



# this stores 1D array of sub images so we have to reshape to recreate image
# for i in range(100):
# for i in range(10000):
restored_image = np.zeros((94, 94), dtype='float64')

reds7_7 = tf.image.extract_patches(
        reds,
        sizes=[1, 7, 7, 1], 
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

# print(f"reds7_7.shape = {reds7_7.shape}") 
# --> 

# for i in range(100):
total_image_num = reds7_7.shape[1] * reds7_7.shape[2] 
reds7_7 = tf.reshape(reds7_7, (total_image_num, 7, 7))

print(f"reds7_7.shape = {reds7_7.shape}")
restored_image = tf.reshape(reds7_7[:, 7//2, 7//2], (94, 94, 1))
print(restored_image.shape)

plt.imshow(restored_image)
plt.show()
# for im in reds7_7:


# print(images_7x7.shape)
# n = 10
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
# images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

# We generate two outputs as follows:
# 1. 3x3 patches with stride length 5
# 2. Same as above, but the rate is increased to 2
# images = np.array(images)
# # print(images.shape)
# im_patche = tf.image.extract_patches(images=images,
#                            sizes=[1, 5, 5, 1],
#                            strides=[1, 5, 5, 1],
#                            rates=[1, 1, 1, 1],
#                            padding='VALID')

# 0v0  
# # for i in range n:
# #     for j in range n:
# pprint(images)

# print(im_patche.shape)



# print(f"images.shape = {images.shape}")
