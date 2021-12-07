from config import *

from matplotlib import pyplot as plt 
import numpy as np
import tensorflow as tf
import os
import shutil




class KirchhoffPD:
    def __init__(self, material, plate_config, sim_config, load_config, bc_config):
        self.material = material
        self.plate_config = plate_config
        self.sim_config = sim_config
        self.load_config = load_config
        self.bc_config = bc_config

        # initial displacement
        self.disp_z = tf.Variable(self.bc_config.disp_BC)

        # initial coordinate
        self.init_coord = self.prepare_coord_array()



        # print(self.disp_z)

    def clear_output(self, output_dir='output'):
        """ clear output directory and create new directory"""

        try:
            os.mkdir(output_dir)
        except OSError as error:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        os.mkdir(os.path.join(output_dir, 'output_whole_steps'))


    def summary(self):
        self.material.summary()
        self.plate_config.summary()
        self.sim_config.summary()


    def run_static(safety_factor=0.5, output_dir="output"):
        self.clear_output(output_dir=output_dir)
        # initital output
        # output_one_step()
        for step in range():
            pd_forces = calc_PD_force()

            self.disp_z = explicit_time_integrate(self.disp_z)
            
            self.disp_z[mask] = disp_BC[mask]

            # output_one_step()

    def plot_2D_disp(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        ax.imshow(self.disp_z)
        plt.show()


    def plot_nodes(self):
        # xyz = np.reshape(self.init_coord, self.plate_config.)
        xyz = self.init_coord

        fig = plt.figure()
        ax = fig.add_subplot()
        # ax = plt.axes(projection='3d')
        ax.scatter(x=xyz[:, :, 0], y=xyz[:, :, 1] , s=1)
        # ax.scatter(xs=xyz[:, 0], ys=xyz[:, 1], zs=xyz[:, 2])
        ax.set_title("initial coordinate")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')

        plt.show()

    def prepare_coord_array(self):
        """ create initial coord xyz aray: shape=(nrow_ny, ncol_nx, 3dof)"""
        dx = self.plate_config.dx
        nrow = self.plate_config.row_num  # for y num
        ncol = self.plate_config.col_num  # for x num

        x_start = -dx * (ncol - 1) / 2.0
        y_start = -dx * (nrow - 1) / 2.0

        x_stop =   dx * (ncol - 1) / 2.0
        y_stop =   dx * (nrow - 1) / 2.0



        # X,Y = np.mgrid[x_start : x_stop+dx, y_start : y_stop+dx]
        X, Y = np.meshgrid(np.linspace(x_start, x_stop, ncol), np.linspace(y_start, y_stop, nrow))
        Z = np.zeros_like(X)

        xyz = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        xyz_2d = np.reshape(xyz, (nrow, ncol, 3))
        return xyz_2d
        # print(self.bc_config.disp_BC.shape)


if __name__ == '__main__':

    from matplotlib import pyplot as plt 
    import numpy as np

    mat = Material(
            rho = 7850,
            E = 2.0e+11,
            nu= 0.3)

    # mat.summary()


    plate = PlateConfig(
            row_num = 112, # y num
            col_num = 112, # x num
            thickness = 0.01,
            dx = 0.01,
            horizon = 0.01*3
            )

    # plate.summary()


    sim_conf = SimConfig(
            dt = 1,
            total_steps = 1000,
            output_every_xx_steps = 10
            )

    # sim_conf.summary()

    

    load = LoadConfig(plate)
    body_load = np.zeros_like(load.get_body_force())
    body_load[:, 55:57] = 5.0e5 # [N/m^3]
    load.add_body_force(bforce_z=body_load)
    # pressure = np.zeros_like(load.get_body_force())
    # pressure[6:94, 6:94] = 1
    # load.add_pressure(sforce_z=1)
    # load.add_pressure(sforce_z=pressure)
    
    load.plot()



    bc_conf = BCConfig(plate)
    bc_conf.add_dispBC(np.s_[5:7, :], disp_z=0)
    bc_conf.add_dispBC(np.s_[-7:-5, :], disp_z=0)
    bc_conf.add_dispBC(np.s_[:, 5:7], disp_z=0)
    bc_conf.add_dispBC(np.s_[:, -7:-5], disp_z=0)
    bc_conf.plot_dispBC()

    # np.s_[:, 12:7:-1] <- if you want ::-1 then you also have to flip 7&12
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 0:5], source_slicer=np.s_[:, 11:6:-1],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, -1:-6:-1], source_slicer=np.s_[:, -12:-7], BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[0:5, :], source_slicer=np.s_[11:6:-1, :],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[-1:-6:-1, :], source_slicer=np.s_[-12:-7, :], BC_type='simply' )

    bc_conf.plot_Kirchhoff_BC()

    kirchhoff_PD = KirchhoffPD(mat, plate, sim_conf, load, bc_conf)
    kirchhoff_PD.summary()
    # plt.imshow(kirchhoff_PD.init_coord[:, :, 2])
    # plt.show()

    kirchhoff_PD.clear_output()
    # kirchhoff_PD.plot_2D_disp()
    kirchhoff_PD.plot_nodes()


