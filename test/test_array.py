import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# dx = 1
# ncol = 13
# nrow = 13
# x_length = dx * (nrow - 1)
# y_length = dx * (ncol - 1)

# x_start = -x_length / 2.0
# y_start = -y_length / 2.0

# x_stop =  x_length / 2.0
# y_stop =  y_length / 2.0
# # center_x = 0
# # center_y = 0


# X,Y = np.mgrid[x_start : x_stop+dx, y_start : y_stop+dx]
# Z = np.zeros_like(X)

# xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
# xyz_2d = np.reshape(xyz, (nrow, ncol, 3))

# print(xyz.shape)
# print(xyz_2d.shape)
# # plt.imshow(xyz_2d[:, :, 0])
# # plt.imshow(xyz_2d[:, :, 1])
# plt.imshow(xyz_2d[:, :, 2])


# # fig = plt.figure()
# # ax = fig.add_subplot()
# # # ax = plt.axes(projection='3d')
# # ax.scatter(x=xyz[:, 0], y=xyz[:, 1] )
# # # ax.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
# # ax.axis('equal')

# plt.show()

X, Y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
print(X[0:0])

# print(np.arctan2(0, 1))


# arr = np.zeros((1, 3, 3, 1))
# tf_arr = tf.constant(arr)
# print(tf_arr.shape)
# vol = tf.constant([1.5])
# tf_X = tf.Variable(X, dtype='float32')


# print(tf_X[tf.newaxis, :, :] + tf.reshape(tf_X, (25, 1)))
# print(tf_X.shape)
# zeros = tf.zeros((25, 5, 5))
# zeros = tf.zeros((1000, 5, 5, 25))
# ones = tf.ones((1, 5, 5, 25))
# print(zeros + ones)

# print(zeros[:, 2, 2, tf.newaxis, tf.newaxis, :].shape)
def func(a, b):
    a += b

def main():
    a = 1
    b = 2
    print("a = ", a)

    func(a, b)

    print("a = ", a)
    
if __name__=='__main__':

    
    # main()
    # tf.



# ones = tf.ones_like(zeros)
# print(zeros+tf.reshape(ones[:, 1, 1], (1,)))
# one = 1.0
# print(tf.broadcast_to(one, [3, 3]))
# print(tf.broadcast_to(ones[:, 0:1, 0:1], [25, 1, 1]))
# print(ones[:, 1, 1, tf.newaxis, tf.newaxis] + zeros )
# print(zeros+tf.broadcast_to(ones[:, 1, 1], shape=(25, 5, 5)))
# print(zeros + tf.reshape(tf_X, (25, 1 ,1)))

# print(vol + tf_X)
# print(tf.reshape(vol, (2,2)))
# print(tf.reshape(tf_arr, (3, 3)))



