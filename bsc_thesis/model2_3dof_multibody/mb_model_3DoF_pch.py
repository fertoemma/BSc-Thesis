# mb_model_3DoF.py

"""
Solving equation of motion and generating data for training

Important notice: lmfit for k1,k2 and kt_1,kt_2 needs to be implemented

"""

# # Imports
# import os
# import sys 
# sys.path.append(r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling')
# import matplotlib.pyplot as plt
# import function_files.krc_reader as krc_reader
# import function_files.filterSAEJ211 as filter
# import numpy as np
# from lmfit import Parameters, Minimizer
# import model2_3dof_multibody.equation_solver_3DoF as solv
# import function_files.data_saver as data_saver 

# Imports
import os
import sys 
# sys.path.append(r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling')
# megadja a jelenlegi file szülőkönyvtárát
osd_path = os.path.dirname(os.path.dirname(__file__))  # szülőkönyvtár szülőkönyvtára
sys.path.append(osd_path)
print(osd_path)
import matplotlib.pyplot as plt
import function_files.krc_reader as krc_reader
import function_files.filterSAEJ211 as filter
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import RK45
from scipy.interpolate import interp1d
from lmfit import Parameters, Minimizer
import pickle
import sympy as sp
sp.init_printing()
from sympy import symbols, Function, diff, lambdify
from IPython.display import Math
from IPython.display import display
from sklearn.metrics import mean_squared_error
import model2_3dof_multibody.equation_solver_3DoF as solv
import function_files.data_saver as data_saver 
import lmfit_3dof
import lmfit

# Input
base_path = os.path.join(osd_path, 'Dataset')
path_EU2 = os.path.join(base_path, 'EU2')
path_US = os.path.join(base_path, 'US')
sim_repl_lib_paths = [ path_EU2  ]    # EU2 or US dataset
# sim_repl_lib_paths = [ path_US  ]
fname_1 = 'EU2'
krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )
krc_data_reader.load_replications_results( moving_average_win_len=10 )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed() # !!!
simulation_channels_in_data = krc_data_reader.get_available_channels()
simulation_time_vals = krc_data_reader.get_time_values()
# print( channels_in_data )

# Creating dictionary
sim_ids = krc_data_reader.get_available_replications()
# objective function needed
obj_function = lmfit_3dof.combined_obj_function

data_dict = dict()
data_dict_p = dict()    # pelvis
data_dict_ch = dict()   # chest
data_dict_h = dict()    # head
# Symbolic part - Mechanical model 
l = 1.03*1000-180   # mm
R = 9/100*1000           # mm
m_1 = 34        # kg
m_2 = 4         # kg
ddx_num, ddphi_num, ddtheta_num = solv.lagrangian_fast2(m_1, m_2, l, R)   # data can be modified

y0 = np.array ([0,0,0,0,0,0])

# Optuna 500 iterations with only pelvis channels
k_1_opt = 5.076e+05
k_2_opt = 3.377e+06 
kt_1_opt = 3.364e+06
kt_2_opt = 2.542e+07
for key in sim_ids:

    # SLED
    sing_rep_res = krc_data_reader.get_single_replication_results(key)
    sled_data = sing_rep_res['10SLEDFRMI00ACX0'] # SLED
    filtered_data = filter.filterSAEJ211(sled_data, 6, 0.001)
    sled_acceleration_data = np.array(filtered_data) # mm/s^2
    # PELVIS
    pelvis_x_data = sing_rep_res['11PELV0000H3ACX0']
    filtered_pelv_acx_data = filter.filterSAEJ211(pelvis_x_data, 6, 0.001)
    pelvis_acceleration_simu_x = np.array( filtered_pelv_acx_data ) # mm/s^2
    pelvis_y_data = sing_rep_res['11PELV0000H3ACY0']
    filtered_pelv_acy_data = filter.filterSAEJ211(pelvis_y_data, 6, 0.001)
    pelvis_acceleration_simu_y = np.array( filtered_pelv_acy_data ) # mm/s^2
    # CHEST
    chest_x_data = sing_rep_res['11CHST0000H3ACX0']
    filtered_chest_acx_data = filter.filterSAEJ211(chest_x_data, 6, 0.001)
    chest_acceleration_simu_x = np.array( filtered_chest_acx_data ) # mm/s^2
    chest_y_data = sing_rep_res['11CHST0000H3ACY0']
    filtered_chest_acy_data = filter.filterSAEJ211(chest_y_data, 6, 0.001)
    chest_acceleration_simu_y = np.array( filtered_chest_acy_data ) # mm/s^2
    # HEAD
    head_x_data = sing_rep_res['11HEAD0000H3ACX0']
    filtered_head_acx_data = filter.filterSAEJ211(head_x_data, 6, 0.001)
    head_acceleration_simu_x = np.array(filtered_head_acx_data)  # mm/s^2
    head_y_data = sing_rep_res['11HEAD0000H3ACY0']
    filtered_head_acy_data = filter.filterSAEJ211(head_y_data, 6, 0.001)
    head_acceleration_simu_y = np.array(filtered_head_acy_data)  # mm/s^2
    

    # Displacement (gain)
    t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement(simulation_time_vals, sled_acceleration_data)
    gain = -1 * sled_d   # mm, minus sign: sled is pulled backwards, so it is moving in the negative direction in the global CS, but in the datafile, it was logged unsigned.
    
    channels_to_use = { "pelvis" : 1, "chest" : 1, "head" : 1 } # Values: 0 for exclude 1 for include   
  
    #lmfit for optimizing k
    # params=Parameters()
    # # set up the initial parameter values
    # params.add('k_1', min=0, max=None, value=k_1_init_opt, vary=True)  # N/mm
    # params.add('kt_1', min=0, max=None, value=kt_1_init_opt, vary=True)  # Nmm/rad
    # params.add('k_2', min=0, max=None, value=k_2_init_opt, vary=True)  # N/mm
    # params.add('kt_2', min=0, max=None, value=kt_2_init_opt, vary=True)  # Nmm/rad

    # fitter = Minimizer(obj_function, params, 
    #                    fcn_args=(ddx_num, ddphi_num, ddtheta_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0,R, l, channels_to_use))
    # result = fitter.minimize(method = 'least_squares')
   
    # k_1_opt = result.params['k_1']
    # k_2_opt = result.params['k_2']
    # kt_1_opt = result.params['kt_1']
    # kt_2_opt = result.params['kt_2']
    # result.params.pretty_print()

    
    a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, ddtheta_num, gain, simulation_time_vals, y0, k_1_opt, k_2_opt, kt_1_opt, k_2_opt, R, l)

    # Difference
    difference_pelv = solv.difference_calc(pelvis_acceleration_simu_x, a_pelvis_x_combfit) # 4. oszlop
    difference_chest = solv.difference_calc(chest_acceleration_simu_x, a_chest_x_combfit) # 4. oszlop
    difference_head = solv.difference_calc(head_acceleration_simu_x, a_head_x_combfit) # 4. oszlop
    
    tmp_1 = [sled_acceleration_data, a_pelvis_x_combfit, pelvis_acceleration_simu_x, difference_pelv]
    tmp_2 = [sled_acceleration_data, a_chest_x_combfit, chest_acceleration_simu_x, difference_chest]
    tmp_3 = [sled_acceleration_data, a_head_x_combfit, head_acceleration_simu_x, difference_head]
    
    data_dict_p[key] = tmp_1  # ennek kell beadni azt a tmp numpy arrayt amiben a 4x1400 adat van
    data_dict_ch[key] = tmp_2  # ennek kell beadni azt a tmp numpy arrayt amiben a 4x1400 adat van
    data_dict_h[key] = tmp_3  # ennek kell beadni azt a tmp numpy arrayt amiben a 4x1400 adat van
    print("Iteration: ", key, " done.")
# print(data_dict)

# Save 
folder = 'saved_data'
data_saver.save(data_dict_p, folder, 'EU2_11PELV0000H3ACX0_m2_pch')
data_saver.save(data_dict_ch, folder, 'EU2_11CHST0000H3ACX0_m2_pch')
data_saver.save(data_dict_h, folder, 'EU2_11HEAD0000H3ACX0_m2_pch')
