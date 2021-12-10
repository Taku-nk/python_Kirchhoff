from matplotlib import pyplot as plt 
import numpy as np


# just data structure with summary.
class Material:
    def __init__(self, rho=7850.0, E=2.0e+11, nu=0.3):
        """
        Attributes:
            self.density = rho
            self.youngs_modulus = E
            self.poissons_ratio = nu
        """
        self.density = rho
        self.youngs_modulus = E
        self.poissons_ratio = nu


    def summary(self):
        print("Material:")
        print("density         : {}".format(self.density))
        print("Young's modulus : {}".format(self.youngs_modulus))
        print("Poisson's ratio : {}".format(self.poissons_ratio))
        print()



class PlateConfig:
    def __init__(self, row_num, col_num, thickness, dx=1.0, horizon=3.0):
        """ row_num=ynum, col_num=xnum"""
        self.row_num = row_num
        self.col_num = col_num
        self.thickness = thickness
        self.dx = dx
        self.horizon = horizon
        self.vol = dx * dx * thickness



    def summary(self):
        print("Plate Configuration:")
        print("row_num=yn: {}".format(self.row_num))
        print("col_num=xn: {}".format(self.col_num))
        print("thickness : {}".format(self.thickness))
        print("dx        : {}".format(self.dx))
        print("vol       : {}".format(self.vol))
        print("horizon   : {}".format(self.horizon))
        print()



class SimConfig:
    def __init__(self, dt, total_steps=100, output_every_xx_steps=10):
        self.dt = dt
        self.total_steps = total_steps
        self.output_every_xx_steps = output_every_xx_steps

    def summary(self):
        print("Simulation Configuration:")
        print("dt           : {}".format(self.dt))
        print("total steps  : {}".format(self.total_steps))
        print("output every : {} steps".format(self.output_every_xx_steps))
        print()



# here create matrix data

class LoadConfig:
    def __init__(self, plate):
        """
            plate : PlateConfig object
        """
        self.plate = plate
        self.body_force = np.zeros(shape=(self.plate.row_num, self.plate.col_num))

    def add_pressure(self, sforce_z=0):
        """ 
            convert add even pressure(scalar) or any pressure(array like)to body force
            sforce_z : scalar or array like shape = (row_num=yn, col_num=xn)
        """
        # self.body_force += sforce_z / self.plate.thickness
        self.body_force += sforce_z / self.plate.thickness

    def add_body_force(self, bforce_z=0):
        """ 
            add body force even(for scalar), any (for array like)
            bforce_z : scalar or array like shape = (row_num=y_num, col_num=x_num)
        """
        self.body_force += bforce_z

    def get_body_force(self):
        return self.body_force


    def plot(self):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        ax.set_title("Body force")
        ax.grid(ls=':')

        cb = ax.imshow(self.body_force, cmap='viridis')
        plt.colorbar(cb)
        plt.show()



class BCConfig:
    def __init__(self, plate):
        self.plate = plate
        # zero -> no disp BC, one-> have disp BC
        self.disp_BC_mask = np.zeros(shape=(self.plate.row_num, self.plate.col_num), dtype='bool')
        self.disp_BC = np.zeros(shape=(self.plate.row_num, self.plate.col_num))

        # self.disp_BC_list = []

        # slicer list for Kirchhoff BC
        self.Kirchhoff_BC_clamp_list = [] # [(fict slicer1, source slicer1), (fict slicer2, source_slicer3), ...]
        self.Kirchhoff_BC_simply_list = [] # [(fict slicer1, source slicer1), (fict slicer2, source_slicer3), ...]
        

    # def add_dispBC(self, mask, value=np.zeros_like(self.)):
            # mask: np.array of bool, false-> no BC, True-> have BC
    def add_dispBC(self, slicer, disp_z=0):
        """
            slicer : np.s_[::], slicer object or mask (np.array of bool)
        """
        self.disp_BC_mask[slicer] = True
        self.disp_BC[slicer] = disp_z

        # self.disp_BC_list.append((mask, value))

    def plot_dispBC(self):
        

        fig, (ax_value, ax_mask) = plt.subplots(1, 2, figsize=(12,6))
        # fig = plt.figure(figsize=(12, 4))
        # ax_value = fig.add_subplot(211)
        # ax_mask = fig.add_subplot(212)

        # displacement value
        cb_value = ax_value.imshow(self.disp_BC, cmap='viridis')
        # where is the BC 0(False)->no disp BC,   1(True)->have disp BC
        cb_mask = ax_mask.imshow(self.disp_BC_mask, cmap='Greys_r')
        # cb_mask = ax_mask.imshow(self.disp_BC_mask.astype('int'), cmap='Greys_r')

        ax_value.set_title('disp BC')
        ax_mask.set_title('BC mask')


        ax_value.grid(ls=':')
        ax_mask.grid(ls=':')

        fig.colorbar(cb_value, ax=ax_value, shrink=0.7)
        fig.colorbar(cb_mask, ax=ax_mask,   shrink=0.7)

        # plt.tight_layout
        fig.tight_layout
        plt.show()


    def add_Kirchhoff_BC(self, fict_slicer, source_slicer, BC_type='clamped'):
        """ 
            Add Kirchhoff specific boundary condition
            Arguments:
                fict_slicer: np.s_[] (slicer object), fictitious node ID slicer
                source_slicer : np.s_[] (slicer object), source node ID (correcponding to fict node)
                BC_type : str, 'clamped', 'simply' , default is 'clamped'
        """
        if BC_type == 'clamped':
            self.Kirchhoff_BC_clamp_list.append((fict_slicer, source_slicer))
        elif BC_type == 'simply':
            self.Kirchhoff_BC_simply_list.append((fict_slicer, source_slicer))
        else:
            print(f"there is no BC_type = {BC_type}")

        
    def plot_Kirchhoff_BC(self):
        Kir_BC_image = np.zeros((self.plate.row_num, self.plate.col_num))

        for fict_slicer, source_slicer in self.Kirchhoff_BC_clamp_list:
            Kir_BC_image[fict_slicer] = 2
            Kir_BC_image[source_slicer] = 1
            
        for fict_slicer, source_slicer in self.Kirchhoff_BC_simply_list:
            Kir_BC_image[fict_slicer] = 2
            Kir_BC_image[source_slicer] = 1
        # fig, (ax_fict, ax_source) = plt.subplots(1, 2, figsize=(12,6))
        # 0-> no Kir BC, 1->clamp_fict, 2-> clamped_source

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        ax.set_title("Kirchhoff_BC")
        ax.grid(ls=':')

        ax.imshow(Kir_BC_image, cmap='copper')
        plt.show()



if __name__=='__main__':
    from matplotlib import pyplot as plt 
    import numpy as np

    mat = Material(
            rho = 2500,
            E = 7.8e+10,
            nu= 0.28)

    mat.summary()


    plate = PlateConfig(
            row_num = 100,
            col_num = 100,
            thickness = 1,
            dx = 1.0,
            horizon = 3.0
            )

    plate.summary()


    sim_conf = SimConfig(
            dt = 0.01,
            total_steps = 1000,
            output_every_xx_steps = 10
            )

    sim_conf.summary()

    

    load = LoadConfig(plate)
    pressure = np.zeros_like(load.get_body_force())
    pressure[6:94, 6:94] = 1
    # load.add_pressure(sforce_z=1)
    load.add_pressure(sforce_z=pressure)
    load.plot()



    bc_conf = BCConfig(plate)
    bc_conf.add_dispBC(np.s_[5:7, :], disp_z=1)
    bc_conf.add_dispBC(np.s_[-7:-5, :], disp_z=1)
    bc_conf.add_dispBC(np.s_[49:51, 49:51], disp_z=1)
    bc_conf.plot_dispBC()

    # np.s_[:, 12:7:-1] <- if you want ::-1 then you also have to flip 7&12
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 0:5], source_slicer=np.s_[:, 11:6:-1],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 95:100], source_slicer=np.s_[:, 92:87:-1], BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[0:5, :], source_slicer=np.s_[11:6:-1, :],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[95:100, :], source_slicer=np.s_[92:87:-1, :], BC_type='simply' )

    bc_conf.plot_Kirchhoff_BC()
