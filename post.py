import numpy as np

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot()



disp_history = np.load("./output/disp_z_history.npy")
time_history = np.load("./output/time_history.npy")

print(f"disp_history.shape = {disp_history.shape}")
print(f"time_history.shape = {time_history.shape}")

# ax.imshow(disp_history[-1, :, :])

# center coord history
# ax.plot(time_history, disp_history[:, disp_history.shape[1]//2, disp_history.shape[2]//2])


# center line disp

plt.show()

