

    from matplotlib import pyplot as plt 
    import numpy as np

    mat = Material(
            rho = 2500,
            E = 7.8e+10,
            nu= 0.28)

    # mat.summary()


    plate = PlateConfig(
            row_num = 100,
            col_num = 100,
            thickness = 1,
            dx = 1.0,
            horizon = 3.0
            )

    # plate.summary()


    sim_conf = SimConfig(
            dt = 0.01,
            total_steps = 1000,
            output_every_xx_steps = 10
            )

    # sim_conf.summary()

    

    load = LoadConfig(plate)
    pressure = np.zeros_like(load.get_body_force())
    pressure[6:94, 6:94] = 1
    # load.add_pressure(sforce_z=1)
    load.add_pressure(sforce_z=pressure)
    # load.plot()



    bc_conf = BCConfig(plate)
    bc_conf.add_dispBC(np.s_[5:7, :], disp_z=1)
    bc_conf.add_dispBC(np.s_[-7:-5, :], disp_z=1)
    bc_conf.add_dispBC(np.s_[49:51, 49:51], disp_z=1)
    # bc_conf.plot_dispBC()

    # np.s_[:, 12:7:-1] <- if you want ::-1 then you also have to flip 7&12
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 0:5], source_slicer=np.s_[:, 11:6:-1],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[:, 95:100], source_slicer=np.s_[:, 92:87:-1], BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[0:5, :], source_slicer=np.s_[11:6:-1, :],     BC_type='simply' )
    bc_conf.add_Kirchhoff_BC(fict_slicer=np.s_[95:100, :], source_slicer=np.s_[92:87:-1, :], BC_type='simply' )

    # bc_conf.plot_Kirchhoff_BC()

    kirchhoff_PD = KirchhoffPD(mat, plate, sim_conf, load, bc_conf)
    kirchhoff_PD.summary()
    # plt.imshow(kirchhoff_PD.init_coord[:, :, 2])
    # plt.show()

    kirchhoff_PD.clear_output()
