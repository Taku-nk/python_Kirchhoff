import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import animation 

class PlotFigure:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # self.fig = plt.figure(figsize=(12, 8))
        self.fig = plt.figure(figsize=(8, 6))
        self.ax0 = self.fig.add_subplot(221)
        self.ax1 = self.fig.add_subplot(222)
        self.ax2 = self.fig.add_subplot(223)
        self.ax3 = self.fig.add_subplot(224, projection='3d')

        # self.disp_history = np.load("./output/disp_z_history.npy")
        # self.time_history = np.load("./output/time_history.npy")
        # self.init_coord = np.load("./output/init_coord.npy")

        self.disp_history = np.load(os.path.join(self.data_dir, "disp_z_history.npy"))
        self.time_history = np.load(os.path.join(self.data_dir, "time_history.npy"  ))
        self.init_coord   = np.load(os.path.join(self.data_dir, "init_coord.npy"    ))

        self.time_step_idx = -1

    def SetHistoryIndex(self, idx):
        """ Set which time step index is used for visualize the result.
        Default is -1 (which is the last result of the displacement)"""
        self.time_step_idx = idx

    
    def plot_history(self, ax):
        """ Plot disp history at center node """
        ax.set_title('History at center')
        ax.grid(ls=':')
        ax.set_xlabel('Time')
        ax.set_ylabel('z-displacement')
        ax.plot(self.time_history, self.disp_history[:, self.disp_history.shape[1]//2, self.disp_history.shape[2]//2])

    def plot_3D(self, ax):
        """ Plot 3D visualization """

        xs = self.init_coord[:, :, 0].flatten()
        ys = self.init_coord[:, :, 1].flatten()
        zs = self.disp_history[self.time_step_idx, :, :].flatten()


        ax.set_title('Absolute displacement')
        ax.set_box_aspect((1,1,0.5))
        # ax.set_zlim(-0.02, 0.02)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Displacement")

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False


        color_data =  zs

        # c= float or 1D array(len=material point)
        sc = ax.scatter(xs, ys, zs, 
                  # vmin=vmin, vmax=vmax, 
                  # vmin=0, vmax=1.5e+6, 
                   c=color_data, 
                   s=2,
                   cmap=plt.get_cmap('viridis_r'), depthshade=False)
        plt.colorbar(sc, shrink=0.8, pad=0.15)
        # plt.show()


    def plot_2D(self, ax, cmap='viridis_r'):
        """ Plot imshow data """
        ax.set_title("2D visualiation")
        cmap=cmap
        im = ax.imshow(self.disp_history[self.time_step_idx, :, :], cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.axis("off")
        plt.colorbar(im, cax=cax) 

    def plot_center_row(self, ax, save=False):
        """ Plot center row (Only center available for now.)"""
        init_coord_x = self.init_coord[:, :, 0]
        row_mask=  np.s_[self.init_coord.shape[0]//2, :]
        ax.set_title("Displacement at center line")
        ax.set_xlabel("Coordinate")
        ax.set_ylabel("z-displacement")
        ax.grid(ls=":")
        ax.plot(init_coord_x[row_mask], self.disp_history[self.time_step_idx, :, :][row_mask],
                marker="o", markersize=2)

        if save == True:
            save_data_as_CSV(self.data_dir, 
                    init_coord_x[row_mask], self.disp_history[self.time_step_idx, :, :][row_mask],
                    xlabel="x", ylabel="z", file_name="row_x1.csv")

    def plot(self, save_center=False, cmap='viridis_r'):
        """ Plot everything in one figure and show """
        self.plot_history(self.ax0)
        self.plot_3D(self.ax3)
        self.plot_2D(self.ax2, cmap=cmap)
        self.plot_center_row(self.ax1, save=save_center)

        self.fig.suptitle("Result at timestep {}".format(self.time_history[self.time_step_idx]))

        plt.tight_layout()
        plt.show()
        pass


def save_data_as_CSV(data_dir, xs, ys, xlabel="x", ylabel="y", file_name="row_x.csv"):
    """ 
    Save plot data as csv. 
    (First create Pandas dataframe, then save to csv)
    """
    df = pd.DataFrame({xlabel:xs, ylabel:ys})
    df.to_csv(os.path.join(data_dir, file_name), index=False )

    print(f"Saved file \"{file_name}\" to \"{os.path.join(data_dir, file_name)}\"")
    

    







if __name__=="__main__":
    data_dir = "./output/"
    plot_result = PlotFigure(data_dir)
    plot_result.plot()
    


    # save_data_as_CSV(data_dir, xs, ys, xlabel="x[m]", ylabel="z[m]")
# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt


# fig = plt.figure(figsize=(4, 12))
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)

# ax1.grid(ls=':')
# ax2.grid(ls=':')
# ax3.grid(ls=':')



# disp_history = np.load("./output/disp_z_history.npy")
# time_history = np.load("./output/time_history.npy")
# init_coord = np.load("./output/init_coord.npy")


# last_disp_z = disp_history[-1]
# init_coord_x = init_coord[:, :, 0]

# # shape of the result matrix
# print(f"disp_history.shape = {disp_history.shape}")
# print(f"time_history.shape = {time_history.shape}")
# print(f"init_coord.shape = {init_coord.shape}")



# # row(y), col(x), dof 0:x, 1:y, 2:z
# # mask to select material points along x center
# row_mask=  np.s_[init_coord.shape[0]//2, :]



# # plot result
# ax1.plot(time_history, disp_history[:, disp_history.shape[1]//2, disp_history.shape[2]//2])
# ax2.plot(init_coord_x[row_mask], last_disp_z[row_mask])
# ax3.imshow(disp_history[-1, :, :])



# # output result to csv
# df = pd.DataFrame({'ini_x':init_coord_x[row_mask], 'disp_z':last_disp_z[row_mask]})
# df.to_csv('./output/row_x.csv', index=False)




# plt.show()

