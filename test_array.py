import numpy as np
from matplotlib import pyplot as plt

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

X, Y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-5, 5, 11))
print(X)
