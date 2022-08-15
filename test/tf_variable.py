import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force CPU to calculate by hiding GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# override existing tf.Variable
# np_arr = np.array([
#                 [1, 1, 1], 
#                 [1, 1, 1],
#                 ])
# print(f"np_arr = \n{np_arr}")

# tf_tensor = tf.Tens([
tf_tensor = tf.Variable([
                [1, 1, 1], 
                [1, 1, 1],
                ], dtype=tf.int32)

print(f"tf_tensor = {tf_tensor}")

tf_zeros = tf.zeros_like(tf_tensor)

tf_mask = tf.constant([
                    [False, False, False],
                    [True, True, False]])


# print(tf_const[tf_mask])
# tf_tensor[tf_mask] = tf_const[tf_mask]

# x=new value , y = base balue
tf_tensor = tf.where(tf_mask, x=tf_zeros, y=tf_tensor)

print(tf_tensor)

# print(tf.where(tf_mask, tf_zeros, tf_tensor))

# print(tf_zeros)


# print(tf.where)

# print(tf_tensor[tf_mask])

# print(f"tf_tensor = {tf_tensor}")

