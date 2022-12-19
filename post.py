import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


fig = plt.figure(figsize=(4, 12))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.grid(ls=':')
ax2.grid(ls=':')
ax3.grid(ls=':')



disp_history = np.load("./output/disp_z_history.npy")
time_history = np.load("./output/time_history.npy")
init_coord = np.load("./output/init_coord.npy")


last_disp_z = disp_history[-1]
init_coord_x = init_coord[:, :, 0]

# shape of the result matrix
print(f"disp_history.shape = {disp_history.shape}")
print(f"time_history.shape = {time_history.shape}")
print(f"init_coord.shape = {init_coord.shape}")



# row(y), col(x), dof 0:x, 1:y, 2:z
# mask to select material points along x center
row_mask=  np.s_[init_coord.shape[0]//2, :]



# plot result
ax1.plot(time_history, disp_history[:, disp_history.shape[1]//2, disp_history.shape[2]//2])
ax2.plot(init_coord_x[row_mask], last_disp_z[row_mask])
ax3.imshow(disp_history[-1, :, :])



# output result to csv
df = pd.DataFrame({'ini_x':init_coord_x[row_mask], 'disp_z':last_disp_z[row_mask]})
df.to_csv('./output/row_x.csv', index=False)




plt.show()

