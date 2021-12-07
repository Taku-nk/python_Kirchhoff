import numpy as np
import matplotlib.pyplot as plt

from matplotlib import image

image = image.imread('./100_100.png')
print(image.dtype)
print(image.shape)

red = image[:,:,0]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
cb = ax.imshow(red, cmap='turbo')

fig.colorbar(cb, ax=ax)
# red = image[:,:,0:1]
# print(red.shape)

# plt.imshow(image)
# plt.imshow(red, cmap='turbo')
plt.show()
