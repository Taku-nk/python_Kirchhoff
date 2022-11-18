from config import *

from matplotlib import pyplot as plt 
import numpy as np
import tensorflow as tf
import os
import shutil

from math import e

from time import time




class KirchhoffPD:
    def __init__(self, material, plate_config, sim_config, load_config, bc_config):
        self.material = material
        self.plate_config = plate_config
        self.sim_config = sim_config
        self.load_config = load_config
        self.bc_config = bc_config

        # initial displacement
        self.disp_z = tf.Variable(self.bc_config.disp_BC, dtype=tf.float32)

        # initial coordinate
        # self.init_coord = self.prepare_coord_array()
        self.init_coord = tf.constant(self.prepare_coord_array(), dtype=tf.float32)

        self.KERNEL_SIZE = self.calc_kernel_size()

        self.disp_output_list = []
        self.time_output_list = []


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


    def run_dynamic(self, output_dir="output"):
        """
        explicit dynamic 
        """
        # KERNEL_SIZE = self.calc_kernel_size()
        dx = self.plate_config.dx
        h = self.plate_config.thickness
        PI = np.pi
        vol = self.plate_config.vol
        body_forces = self.load_config.body_force
        time = tf.Variable(0.0, dtype=tf.float32)
        dt = tf.constant(self.sim_config.dt, dtype=tf.float32)

         # print("vol = ", vol)
        # print(PI)

        # print("Kernel size = ", KERNEL_SIZE)
        self.clear_output(output_dir=output_dir)
        # initital output
        # output_one_step()
        # print(self.sim_config.total_steps)
        disp_z_old = tf.Variable(np.zeros_like(self.disp_z), dtype=tf.float32)

        self.disp_output_list.append(self.disp_z)
        self.time_output_list.append(time)

        # for time_step in range(10):
        for time_step in range(self.sim_config.total_steps):
            time = time + dt


            pd_forces = self.calc_PD_force()

            # self.disp_z, disp_z_old, velhalf, velhalf_old, pd_forces, pd_forces_old = \
            self.disp_z, disp_z_old = self.explicit_time_integration(
                                                self.disp_z, disp_z_old,
                                                pd_forces, body_forces,
                                                time_step, dt)


            self.apply_disp_BC()
            self.apply_kirchhoff_BC()


            print(f"step {time_step} done (time = {time})")

            if ((time_step+1) % (self.sim_config.output_every_xx_steps)) == 0:
                print(f"output_data, at time = {time}")
                self.disp_output_list.append(self.disp_z)
                self.time_output_list.append(time)

                self.save_numpy_result(output_dir=output_dir)




    def run_static(self, safety_factor=0.5, output_dir="output"):
        # KERNEL_SIZE = self.calc_kernel_size()
        dx = self.plate_config.dx
        h = self.plate_config.thickness
        PI = np.pi
        vol = self.plate_config.vol
        body_forces = self.load_config.body_force
        time = tf.Variable(0.0, dtype=tf.float32)
        dt = tf.constant(self.sim_config.dt, dtype=tf.float32)

         # print("vol = ", vol)
        # print(PI)

        # print("Kernel size = ", KERNEL_SIZE)
        self.clear_output(output_dir=output_dir)
        # initital output
        # output_one_step()
        # print(self.sim_config.total_steps)
        # disp_z_old = tf.Variable(np.zeros_like(self.disp_z), dtype=tf.float32)
        pd_forces_old = tf.Variable(np.zeros_like(body_forces), dtype=tf.float32) 

        velhalf = tf.Variable(np.zeros_like(self.disp_z), dtype=tf.float32)
        velhalf_old = tf.Variable(np.zeros_like(self.disp_z), dtype=tf.float32)
        # print("pd_forces_old = ", pd_forces_old.shape)
        # print("disp_z_old.shape = ", disp_z_old.shape)
        # for step in range(1):
        # for step in range(self.sim_config.total_steps):
        self.disp_output_list.append(self.disp_z)
        self.time_output_list.append(time)

        # for time_step in range(10):
        for time_step in range(self.sim_config.total_steps):
            time = time + dt


            pd_forces = self.calc_PD_force()

            # self.disp_z, disp_z_old, velhalf, velhalf_old, pd_forces, pd_forces_old = \
            self.disp_z, velhalf, velhalf_old, pd_forces, pd_forces_old = \
                    self.ADR_time_integration(
                            self.disp_z, velhalf, velhalf_old,
                            pd_forces, pd_forces_old, body_forces,
                            time_step ,safety_factor, dt)

            # print("velhalf    = ", tf.reduce_sum(velhalf))
            # print("velhalf_old= ", tf.reduce_sum(velhalf_old))

            self.apply_disp_BC()
            self.apply_kirchhoff_BC()


            print(f"step {time_step} done (time = {time})")

            if ((time_step+1) % (self.sim_config.output_every_xx_steps)) == 0:
                print(f"output_data, at time = {time}")
                self.disp_output_list.append(self.disp_z)
                self.time_output_list.append(time)

                self.save_numpy_result(output_dir=output_dir)
                            # self.disp_z, disp_z_old, velhalf, velhalf_old,
            # forces = pd_forces + body_forces
            # return forces
            # print(body_force.shape + )
            # print(pd_forces.shape)
            # return pd_forces
            # forces = 
            # self.disp_z = self.explicit_time_integrate(self.disp_z, pd_forces)

            # self.save_numpy_result(output_dir=output_dir)

            # self.disp_z[mask] = disp_BC[mask]
            # self.apply_disp_BC()
            # self.apply_kirchhoff_BC()


    def save_numpy_result(self, output_dir=''):
        disp_history = np.array(self.disp_output_list)
        time_history = np.array(self.time_output_list)
        init_coord = np.array(self.init_coord)

        # np.savetxt("disp_z_history.csv", disp_history, delimiter=',')
        # np.savetxt("time_history.csv", time_history, delimiter=',')
        # np.save('disp_z_history.npy', disp_history)
        # np.save('time_history.npy', time_history)
        np.save(os.path.join(output_dir,'disp_z_history.npy'), disp_history)
        np.save(os.path.join(output_dir,'time_history.npy'), time_history)
        np.save(os.path.join(output_dir,'init_coord.npy'), init_coord)

        print("saved numpy data at {}".format(os.path.join(os.getcwd(), output_dir)))

        # print(disp_history.shape)
        # print(disp_history)
        # print(time_history)
        


    def explicit_time_integration(self, disp_z, disp_z_old,
                             pd_forces, body_forces,
                             time_step, dt):
        """
        Arguments
        disp_z: current displacement
        disp_z_old: one step before
        pd_force:
        body_force:
        time_step:
        dt
        """
        emod   = tf.constant(self.material.youngs_modulus, dtype=tf.float32)
        pratio = tf.constant(self.material.poissons_ratio, dtype=tf.float32)
        thick  = tf.constant(self.plate_config.thickness, dtype=tf.float32)
        delta  = tf.constant(self.plate_config.horizon, dtype=tf.float32)
        dx     = tf.constant(self.plate_config.dx, dtype=tf.float32)
        dens = tf.constant(self.material.density, dtype=tf.float32)

        # print(f"{thick:g}")
        
        PI = tf.constant(np.pi, dtype=tf.float32)

        # explicit time integration
        disp_z_new = dt * dt * (pd_forces + body_forces) / dens + 2.0 * disp_z - disp_z_old

        # disp_z_old = disp_z 
        
        return disp_z_new, disp_z



    # def ADR_time_integration(self, disp_z, disp_z_old, velhalf, velhalf_old,
    def ADR_time_integration(self, disp_z, velhalf, velhalf_old,
                             pd_forces, pd_forces_old, body_forces,
                             time_step, safety_factor, dt):
        """ 
            Adaptive Dynamic Relaxation time integration for static problem
            Arguments
            disp:
            # disp_old:

            # velhalf:
            velhalf_old:

            pd_force:
            pd_force_old:
            
            time_step
            safety_factor
            dt:

            Returns:
            disp:
            disp_old:

            velhalf:
            velhalf_old

            pd_force:
            pd_force_old:
        """
        emod   = tf.constant(self.material.youngs_modulus, dtype=tf.float32)
        pratio = tf.constant(self.material.poissons_ratio, dtype=tf.float32)
        thick  = tf.constant(self.plate_config.thickness, dtype=tf.float32)
        delta  = tf.constant(self.plate_config.horizon, dtype=tf.float32)
        dx     = tf.constant(self.plate_config.dx, dtype=tf.float32)

        # print(f"{thick:g}")
        
        PI = tf.constant(np.pi, dtype=tf.float32)

        # print("hello!!! = ", PI)
        # print(delta)

        # print("pd_forces     = ", tf.reduce_sum(pd_forces))
        # print("pd_forces_old = ", tf.reduce_sum(pd_forces_old))

        # stable mass vector for ADR
        para_SMV1 = (3.0 * emod) / ((PI * delta **4) * (1.0 + pratio))

        massvec = 0.25 * dt**2 * ((4.0 * para_SMV1 * PI * thick * delta**2) / dx) * safety_factor 

        cn = 0.0
        cn1 = 0.0
        cn2 = 0.0

        # velhalf_old zero mask(zero-->False, else-->True)
        cn1 = tf.reduce_sum(
                -disp_z[velhalf_old!=0.0]**2 * \
                (pd_forces[velhalf_old!=0.0] / massvec - \
                pd_forces_old[velhalf_old!=0.0] / massvec) / \
                (dt * velhalf_old[velhalf_old!=0.0])
                )
        # print(f"cn1 = {cn1}", cn1)

        cn2 = tf.reduce_sum(
                disp_z **2
                )


        if cn2 != 0.0:
            if((cn1 / cn2) > 0.0):
                cn = 2.0 * tf.math.sqrt(cn1 / cn2)
            else:
                cn = 0.0
        else:
            cn = 0.0

        if (cn > 2.0):
            cn = 1.9


        # print(cn1)
        # print(cn2)

        if time_step == 0:
            velhalf = (1.0 * (dt / massvec) * (pd_forces + body_forces)) / 2.0
        else:
            velhalf = ((2.0 - cn * dt) * velhalf_old + 2.0 *  (dt / massvec) * \
                    (pd_forces + body_forces)) / (2.0 + cn * dt)

        # print("velhalf     = ", tf.reduce_sum(velhalf))
        # print("velhalf_old = ", tf.reduce_sum(velhalf_old))

        vel = 0.5 * (velhalf_old + velhalf)
        disp_z = disp_z + velhalf * dt




        velhalf_old = velhalf
        pd_forces_old = pd_forces

        return  disp_z, velhalf, velhalf_old, pd_forces, pd_forces_old


    def calc_PD_force(self):
    # def calc_PD_force2(self):
        """
            performance improved by reduces loop nesting depth.
            return internal forces (e.g. shape=(100, 100))
        """
        dx = tf.constant(self.plate_config.dx          , dtype='float32')
        horizon = tf.constant(self.plate_config.horizon, dtype='float32')
        thick = tf.constant(self.plate_config.thickness, dtype='float32')
        PI = tf.constant(np.pi                         , dtype='float32')
        LEN_j= self.KERNEL_SIZE**2

        flectural_rigidity = tf.constant(
                (self.material.youngs_modulus * thick**3) / \
                (12.0 * (1.0-self.material.poissons_ratio**2)), dtype='float32')

        nu = tf.constant(self.material.poissons_ratio  , dtype='float32')
        # every volume is same in linear grid model.
        vol = tf.constant(self.plate_config.vol        , dtype='float32') #scalar

        # smooth factor for j
        smooth_fac_kernel = tf.constant(self.calc_dist_smooth_fac()[np.newaxis, :, :], dtype='float32')
        smooth_fac_kernel = tf.reshape(smooth_fac_kernel, shape=(1, 1, 1, LEN_j))

        # dist^2 for j
        dist_pow2_kernel = tf.constant(self.calc_dist_pow2()[np.newaxis, :, :], dtype='float32')
        dist_pow2_kernel = tf.reshape(dist_pow2_kernel, shape=(1, 1, 1, LEN_j))

        angle_kernel = tf.constant(self.calc_angle()[np.newaxis, :, :, np.newaxis] , dtype='float32')
        angle_kernel = tf.reshape(angle_kernel, shape=(1, 1, 1, LEN_j))



        disp_z_k_j = tf.image.extract_patches(
                        self.disp_z[tf.newaxis, :, :, tf.newaxis],
                        sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1], 
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')  # shape=(1, 112, 112, 49)
        # it ensures the force output size is same whith the original disp size
        # it doesn't affect the result. because it is only applied to fict node, or will be masked out
        # print(disp_z_k_j)



        # disp_z_k_j = tf.reshape(
        #         disp_z_k_j, 
        #         shape=(disp_z_k_j.shape[1] * disp_z_k_j.shape[2], self.KERNEL_SIZE, self.KERNEL_SIZE))
        # shape=(112*112, 7, 7) 
        # print(disp_z_k_j.shape)

        # calculate psi
        psi1 = tf.reduce_sum(
               (disp_z_k_j - 
                disp_z_k_j[:, :, :, LEN_j//2, tf.newaxis]) /\
                dist_pow2_kernel * vol * smooth_fac_kernel 
                , axis=3, keepdims=True)

        psi2 = tf.reduce_sum(
               (disp_z_k_j - 
                disp_z_k_j[:, :, :, LEN_j//2, tf.newaxis]) /\
                dist_pow2_kernel * vol * smooth_fac_kernel *\
                tf.sin(2*angle_kernel)
                , axis=3, keepdims=True)

        psi3 = tf.reduce_sum(
               (disp_z_k_j - 
                disp_z_k_j[:, :, :, LEN_j//2, tf.newaxis]) /\
                dist_pow2_kernel * vol * smooth_fac_kernel *\
                tf.cos(2*angle_kernel)
                , axis=3, keepdims=True)

        # print(disp_z_k_j.shape)
        # print(disp_z_k_j[:, :, :, LEN_j//2, tf.newaxis].shape)


        psi1_k_j = tf.image.extract_patches(psi1, sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')  # shape=(1, 112, 112, 49)
        psi2_k_j = tf.image.extract_patches(psi2, sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')  # shape=(1, 112, 112, 49)
        psi3_k_j = tf.image.extract_patches(psi3, sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')  # shape=(1, 112, 112, 49)

        
        # start = time()

        forces = (8.0 * flectural_rigidity) / (PI**2 * horizon**4 * thick**3) * \
                tf.reduce_sum(\

                    1.0 / dist_pow2_kernel * \
                    ( \
                    (1.0 + nu) * \
                    (psi1_k_j[:, :, :, LEN_j//2, tf.newaxis] - psi1_k_j) + \
                    4.0 * (1.0 - nu) * (\
                    (psi2_k_j[:, :, :, LEN_j//2, tf.newaxis] - psi2_k_j) * tf.sin(2*angle_kernel) + \
                    (psi3_k_j[:, :, :, LEN_j//2, tf.newaxis] - psi3_k_j) * tf.cos(2*angle_kernel)
                    )) *\
                    vol * smooth_fac_kernel \
                    , axis=3, keepdims=True)
        
        # print(f"Time: {time() - start}")
        # print(forces.shape)

        # print(psi1)
        return tf.reshape(forces, shape=[self.plate_config.row_num, self.plate_config.col_num])
            

    # def calc_PD_force(self):
    def calc_PD_force2(self):
        """ 
            return internal forces (e.g. shape=(100,100))
        """
        # KERNEL_SIZE = self.calc_kernel_size()
        # constant value in this simulation
        dx = tf.constant(self.plate_config.dx          , dtype='float32')
        horizon = tf.constant(self.plate_config.horizon, dtype='float32')
        thick = tf.constant(self.plate_config.thickness, dtype='float32')
        PI = tf.constant(np.pi                         , dtype='float32')


        # length of loop index i_k, i_j
        LEN_i = self.KERNEL_SIZE **2
        KERNEL_SIZE = self.KERNEL_SIZE

        flectural_rigidity = tf.constant(
                (self.material.youngs_modulus * thick**3) / \
                (12.0 * (1.0-self.material.poissons_ratio**2)), dtype='float32')

        # print("flectural_rigidity= {:g}".format(flectural_rigidity))
        # poisson's ratio
        nu = tf.constant(self.material.poissons_ratio  , dtype='float32')
        # every volume is same in linear grid model.
        vol = tf.constant(self.plate_config.vol        , dtype='float32')

        # smooth factor for j
        # smooth factor for i_k, i_j
        smooth_fac_kernel = tf.constant(self.calc_dist_smooth_fac()[np.newaxis, :, :, np.newaxis], dtype='float32')
        smooth_fac_i = tf.reshape(smooth_fac_kernel, shape=(1, 1, 1, LEN_i))

        # print("smooth sum = ", vol * tf.reduce_sum(smooth_fac_kernel))
        # print("analy = ", horizon ** 2 * PI * thick)
        # angle for j
        # angle for i_k, i_j
        angle_kernel = tf.constant(self.calc_angle()[np.newaxis, :, :, np.newaxis] , dtype='float32')
        angle_i = tf.reshape(angle_kernel, shape=(1, 1, 1, LEN_i))

        # dist^2 for j
        # dist^2 for i_k, i_j
        dist_pow2_kernel = tf.constant(self.calc_dist_pow2()[np.newaxis, :, :, np.newaxis], dtype='float32')
        dist_pow2_i = tf.reshape(dist_pow2_kernel, shape=(1, 1, 1, LEN_i))



        # start = time()
        disp_z_k_j = tf.image.extract_patches(
                        self.disp_z[tf.newaxis, :, :, tf.newaxis],
                        sizes=[1, self.KERNEL_SIZE*2-1, self.KERNEL_SIZE*2-1, 1], 
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        # padding='VALID')
                        padding='SAME') 
        # print(f"Time: {time() - start}")
        # it ensures the force output size is same whith the original disp size
        # it doesn't affect the result. because it is only applied to fict node, or will be masked out

        disp_z_k_j = tf.reshape(
                disp_z_k_j, 
                shape=(disp_z_k_j.shape[1] * disp_z_k_j.shape[2], self.KERNEL_SIZE*2-1, self.KERNEL_SIZE*2-1, 1))

        # print("shape disp _z_k_j", disp_z_k_j.shape)


        disp_z_k_j_i = tf.cast(
                     tf.image.extract_patches(
                    disp_z_k_j,
                    sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ), tf.float32)

        # print("shape disp _z_k_j_i", disp_z_k_j_i.shape)

        # forces = disp_z_k_j_i[:, :, :, KERNEL_SIZE//2, tf.newaxis]

        # forces = tf.cast(disp_z_k_j_i[:, KERNEL_SIZE//2, KERNEL_SIZE//2, tf.newaxis, tf.newaxis, :], tf.float32) -  \
        # tf.cast(disp_z_k_j_i[:, KERNEL_SIZE//2, KERNEL_SIZE//2,  tf.newaxis, tf.newaxis, tf.newaxis, LEN_i//2], tf.float32)
        # forces = angle_i

        forces = (8.0 * flectural_rigidity) / (PI**2 * horizon**4 * thick**3) * \
                tf.reduce_sum( # sum j (y direction)
                tf.reduce_sum( # sum j (x direction)
                    1.0 / dist_pow2_kernel * (\

                     tf.reduce_sum(
                         ((disp_z_k_j_i[:, KERNEL_SIZE//2, KERNEL_SIZE//2, tf.newaxis, tf.newaxis, :] - \
                           disp_z_k_j_i[:, KERNEL_SIZE//2, KERNEL_SIZE//2,  tf.newaxis, tf.newaxis, tf.newaxis, LEN_i//2])/\
                           dist_pow2_i)*\
                           (2 + (1 - nu) * (8 * tf.math.cos(angle_i-angle_kernel)**2 - 5))* \
                          vol * smooth_fac_i, axis=3, keepdims=True) \

                    -tf.reduce_sum(
                         ((disp_z_k_j_i - \
                           disp_z_k_j_i[:, :, :, LEN_i//2, tf.newaxis])/\
                           dist_pow2_i)*\
                           (2 + (1 - nu) * (8 * tf.math.cos(angle_i-angle_kernel)**2 - 5))* \
                          vol * smooth_fac_i, axis=3, keepdims=True)\

                ) * vol * smooth_fac_kernel, axis=2, keepdims=True), axis=1, keepdims=True)
        # force.shape= (10000, 1, 1, 1)

        return tf.reshape(forces, shape=[self.plate_config.row_num, self.plate_config.col_num])
        # return forces






    

        # disp_z_k_j_i = tf.
    def PDDO_2D(self):
        """
           implement peridynamic differential opperator 
        """
        ########################################################################
        # constants
        ########################################################################
        dx = tf.constant(self.plate_config.dx          , dtype='float32')
        horizon = tf.constant(self.plate_config.horizon, dtype='float32')
        thick = tf.constant(self.plate_config.thickness, dtype='float32')
        PI = tf.constant(np.pi                         , dtype='float32')
        LEN_j= self.KERNEL_SIZE**2

        vol = tf.constant(self.plate_config.vol        , dtype='float32') #scalar
    
        # smooth factor for j
        smooth_fac_kernel = tf.constant(self.calc_dist_smooth_fac()[np.newaxis, :, :], dtype='float32')
        smooth_fac_kernel = tf.reshape(smooth_fac_kernel, shape=(1, 1, 1, LEN_j))


        # dist^2 for j
        dist_pow2_kernel = tf.constant(self.calc_dist_pow2()[np.newaxis, :, :], dtype='float32')
        dist_pow2_kernel = tf.reshape(dist_pow2_kernel, shape=(1, 1, 1, LEN_j))

        #  local coord xi
        # you have to check the local coordinate direction. it might cause problems
        hor_coord_kernel = self.calc_horizon_coord()
        xi_x_kernel = tf.reshape(tf.constant(hor_coord_kernel[:, 0], dtype='float32'), shape=(1, 1, 1, LEN_j))
        xi_y_kernel = tf.reshape(tf.constant(hor_coord_kernel[:, 1], dtype='float32'), shape=(1, 1, 1, LEN_j))
        print(xi_y_kernel)

        # angle
        angle_kernel = tf.constant(self.calc_angle()[np.newaxis, :, :, np.newaxis] , dtype='float32')
        angle_kernel = tf.reshape(angle_kernel, shape=(1, 1, 1, LEN_j))

        ########################################################################
        ########################################################################

        # global coord 
        x = self.init_coord[:, :, 0]
        y = self.init_coord[:, :, 1]

        # functions to analyze (dummy)
        f = x**2 + y**2

        # print(type(f))


        f_kj = tf.image.extract_patches(
                        f[tf.newaxis, :, :, tf.newaxis],
                        sizes=[1, self.KERNEL_SIZE, self.KERNEL_SIZE, 1], 
                        strides=[1, 1, 1, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')  # shape=(1, 112, 112, 49)

        print(f_kj.shape)
        # plt.imshow(tf.reshape(smooth_fac_kernel, shape=(7,7,1)))
        # plt.imshow(tf.reshape(f_kj[0, 0, 0, :], shape=(7, 7, 1)))
         

        plt.show()

        # psi1 = tf.reduce_sum(
        #        (f_kj -
        #         f_kj[:, :, :, LEN_j//2, tf.newaxis]) /\
        #         dist_pow2_kernel * vol * smooth_fac_kernel 
        #         , axis=3, keepdims=True)


        # plt.imshow(f) 
        # plt.show()
        

    def weight_func_w(self, shape=(1, 1, 1, 7*7)):
        """ return gamma weight function  default shape == (1, 1, 1, 7^2) """
        LEN_j= self.KERNEL_SIZE**2
        horizon = tf.constant(self.plate_config.horizon, dtype='float32')

        dist_pow2_kernel = tf.constant(self.calc_dist_pow2()[np.newaxis, :, :], dtype='float32')
        dist_pow2_kernel = tf.reshape(dist_pow2_kernel, shape=(1, 1, 1, LEN_j))
        # dist_kernel      = tf.sqrt(dist_pow2_kernel)

        weight = e ** (- 4.0 * dist_pow2_kernel / horizon**2)

        # print(weight.shape)
        # plt.imshow(tf.reshape(weight, shape=(7,7)))
        # plt.show()

        return weight


    def plot_3D(self, nodes):
        """
        Arguments
            nodes : np.array, size=(n, 3) (n nodes, 3D)
        """
        pass



        # pass

    def calc_kernel_size(self):
        return int(self.plate_config.horizon / self.plate_config.dx) * 2 + 1
        # return (self.plate_config.horizon // self.plate_config.dx) * 2 + 1

    
    def calc_dist_pow2(self):
        """ calculate xi^2 <- greek xi distance between two nodes)
            center node's value is force make it '1e-38' to prevent zero division
        """
        dx = self.plate_config.dx
        start = - dx * (self.KERNEL_SIZE - 1) / 2.0
        stop  =   dx * (self.KERNEL_SIZE - 1) / 2.0

        X, Y = np.meshgrid(np.linspace(start, stop, self.KERNEL_SIZE), np.linspace(start, stop, self.KERNEL_SIZE))
        xy = np.vstack((X.flatten(), Y.flatten())).T
        xy_2d = np.reshape(xy, (self.KERNEL_SIZE, self.KERNEL_SIZE, 2))

        dist_pow2 = np.sum(xy_2d**2, axis=-1)
        dist_pow2[self.KERNEL_SIZE//2, self.KERNEL_SIZE//2] = 1e-10 # prevent zero division
        return dist_pow2


    def calc_horizon_coord(self):
        """
        return xi_x, xi_y (distance between two nodes (in a family member))
        shape=(KERNEL_SIZE^2, 2) <- 2 for x and y
        returns stack of xi_x and xi_y.
        """
        dx = self.plate_config.dx
        start = - dx * (self.KERNEL_SIZE - 1) / 2.0
        stop  =   dx * (self.KERNEL_SIZE - 1) / 2.0

        X, Y = np.meshgrid(np.linspace(start, stop, self.KERNEL_SIZE), np.linspace(start, stop, self.KERNEL_SIZE))
        xy = np.vstack((X.flatten(), Y.flatten())).T
        # xy_2d = np.reshape(xy, (self.KERNEL_SIZE, self.KERNEL_SIZE, 2))
        return xy




    def calc_dist_smooth_fac(self):
        """
            volume correction factor and mask based on the distance between two node
            returns: 7 x 7 KERNEL factor
            volume correction factor and circular mask and center value is zero, so
            it can be used to exclude center material point in loop

        """

        dx = self.plate_config.dx
        horizon = self.plate_config.horizon

        vol = self.plate_config.vol
        thick  = self.plate_config.thickness




        dist_kernel = np.sqrt(self.calc_dist_pow2())

        factor = np.zeros_like(dist_kernel)
        factor[dist_kernel < horizon + dx/2.0] = (horizon + dx/2.0 - dist_kernel[dist_kernel < horizon + dx/2.0]) / dx
        factor[dist_kernel < horizon - dx/2.0] = 1.0
        factor[self.KERNEL_SIZE//2, self.KERNEL_SIZE//2] = 0.0


        vol_cor_fac = (thick * horizon**2 * np.pi) / np.sum(vol * factor)


        
        # print(dist_kernel < horizon - dx/2.0)

        # return factor
        return factor * vol_cor_fac
        # return factor 




    def calc_angle(self):
        dx = self.plate_config.dx
        start = - dx * (self.KERNEL_SIZE - 1) / 2.0
        stop  =   dx * (self.KERNEL_SIZE - 1) / 2.0

        X, Y = np.meshgrid(np.linspace(start, stop, self.KERNEL_SIZE), np.linspace(start, stop, self.KERNEL_SIZE))
        xy = np.vstack((X.flatten(), Y.flatten())).T
        xy_2d = np.reshape(xy, (self.KERNEL_SIZE, self.KERNEL_SIZE, 2))

        # this Y-> X order is important
        angle = np.arctan2(Y, X)
        # return (angle / np.pi * 180).round(2)
        # print("angle calculated by fliped y. np.array of Y is arraynged in assending way")
        return angle

        # return Y[::-1]




            # output_one_step()
    # def explicit_time_integrate(self):
    #     pass

    def apply_disp_BC(self):
        """ 
            apply displacement boundary condition 
            only on specified in self.bc_config.disp_BC_mask=True
            TODO slicer base like Kirchhoff BC  is more reliable
        """
        # use tf.where x=new_value, y=base_value
        # mask = tf.constant(self.bc_config.disp_BC_mask)
        # self.disp_z = tf.where(mask, 
        self.disp_z = tf.where(self.bc_config.disp_BC_mask, 
                                x=self.bc_config.disp_BC, 
                                y=self.disp_z)

        # # some how tf.tensor wont work this masking, so convert to numpy temporalily
        # mask = self.bc_config.disp_BC_mask
        # np_disp = np.array(self.disp_z)
        # np_disp[mask] = self.bc_config.disp_BC[mask] 
        # self.disp_z = tf.Variable(np_disp, dtype=tf.float32)

        

    def apply_kirchhoff_BC(self):
        """ apply kirchhoff Boundary condition. mirror or symmetry"""

        for fict_slicer, source_slicer in self.bc_config.Kirchhoff_BC_clamp_list:
            np_disp = np.array(self.disp_z)
            np_disp[fict_slicer] = np_disp[source_slicer]
            self.disp_z = tf.Variable(np_disp, dtype=tf.float32)
            # print(fict_slicer)
            # self.disp_z[fict_slicer] = self.disp_z[source_slicer]

        for fict_slicer, source_slicer in self.bc_config.Kirchhoff_BC_simply_list:
            np_disp = np.array(self.disp_z)
            np_disp[fict_slicer] = -np_disp[source_slicer]
            self.disp_z = tf.Variable(np_disp, dtype=tf.float32)
            
            # print(fict_slicer)
            # self.disp_z[fict_slicer] = -self.disp_z[source_slicer]
            # print(self.disp_z[fict_slicer])
            # flipped_source = -self.disp_z[source_slicer]
            # self.disp_z[fict_slicer] = flipped_source
            # self.disp_z[fict_slicer] = self.disp_z[source_slicer]




    def output_one_step(self):
        """ output traditional cpp style output"""
        pass

    def plot_2D_disp(self):
        """ plot self.disp_z """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        cb = ax.imshow(self.disp_z, cmap='viridis')
        plt.colorbar(cb, ax=ax)
        plt.show()

    # def plot_row(self, coord_slicer, value_slicer, **kwargs):
    # def plot_row(self,
    #         coord_slicer=np.s_[self.plate.row_num//2, :, 0],
    #         value_slicer=np.s_[self.plate.row_num//2, :] , **kwargs):
    # def plot_row(self, coord_slicer, value_slicer):
    def plot_row(self, coord_slicer, value_slicer, **kwargs):
        """ plot row disp (for now)"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        ax.plot(self.init_coord[coord_slicer], self.disp_z[value_slicer], **kwargs)
        ax.grid(ls=':')
        ax.legend()
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

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from matplotlib import pyplot as plt 
    import numpy as np

    mat = Material(
            rho = 7850,
            E = 2.0e+11,
            nu= 0.3)

    # mat.summary()
    dx = 0.01
    thickness = 0.01
    horizon = dx * 3.75


    plate = PlateConfig(
            row_num = 112, # y num
            col_num = 112, # x num
            thickness = thickness,
            dx = dx,
            horizon = horizon
            # horizon = 0.01*3.606
            # horizon = 0.01*3.606
            # horizon = 0.01*4.606
            # horizon = 0.01*5.606
            # horizon = 0.01*6.606
            # horizon = 0.01*7.606
            )

    # plate.summary()


    sim_conf = SimConfig(
            dt = 1,
            total_steps = 100,
            output_every_xx_steps = 100
            )

    # sim_conf.summary()

    

    load = LoadConfig(plate)
    body_load = np.zeros_like(load.get_body_force())
    body_load[:, 55:57] = -5.0e5 # [N/m^3]
    # load.add_body_force(bforce_z=body_load)
    # pressure = np.zeros_like(load.get_body_force())
    # pressure[6:94, 6:94] = 1
    load.add_pressure(sforce_z=-1)
    # load.add_pressure(sforce_z=pressure)
    



    bc_conf = BCConfig(plate)

    # bc_conf.add_dispBC(np.s_[5:7, :], disp_z=0)
    # bc_conf.add_dispBC(np.s_[-7:-5, :], disp_z=0)
    # bc_conf.add_dispBC(np.s_[:, 5:7], disp_z=0)
    # bc_conf.add_dispBC(np.s_[:, -7:-5], disp_z=0)

    # np.s_[:, 12:7:-1] <- if you want ::-1 then you also have to flip 7&12
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 0:5], source_slicer=np.s_[:, 11:6:-1],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, -1:-6:-1], source_slicer=np.s_[:, -12:-7], BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[0:5, :], source_slicer=np.s_[11:6:-1, :],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[-1:-6:-1, :], source_slicer=np.s_[-12:-7, :], BC_type='simply' )


    kirchhoff_PD = KirchhoffPD(mat, plate, sim_conf, load, bc_conf)
    # kirchhoff_PD.summary()

    # kirchhoff_PD.clear_output()


    # load.plot()
    # bc_conf.plot_dispBC()
    # bc_conf.plot_Kirchhoff_BC()

    # kirchhoff_PD.plot_nodes()
    
    
    # plt.imshow(kirchhoff_PD.run_static(safety_factor=0.3, output_dir="output"))
    # print( kirchhoff_PD.calc_PD_force().shape )

    

    # kirchhoff_PD.PDDO_2D()
    kirchhoff_PD.weight_func_w()

    
    # kirchhoff_PD.run_static(safety_factor=0.3, output_dir="output")





