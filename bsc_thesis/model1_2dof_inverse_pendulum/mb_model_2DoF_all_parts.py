""""
" Itt lehet a for ciklusban még paramétert illeszteni lmfittel, így nem 1 adatsorra fog alapozni"""

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
import model1_2dof_inverse_pendulum.equation_solver_2DoF as solv
import function_files.data_saver as data_saver 
import lmfit_2dof
import lmfit

# Input
# base_path = r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling\Dataset'
base_path = os.path.join(osd_path, 'Dataset')
path_EU = os.path.join(base_path, 'EU2')
path_US = os.path.join(base_path, 'US')
# sim_repl_lib_paths = [ path_US  ]
sim_repl_lib_paths = [ path_EU  ]
krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )
krc_data_reader.load_replications_results( moving_average_win_len=10 )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed() # !!!
channels_in_data = krc_data_reader.get_available_channels()
simulation_time_vals = krc_data_reader.get_time_values()

# Preparing lmfit
# objective function needed
obj_function = lmfit_2dof.combined_obj_function

# Creating dictionary
sim_ids = krc_data_reader.get_available_replications()
data_dict_p = dict()    # pelvis
data_dict_ch = dict()   # chest
data_dict_h = dict()    # head

#  Symbolic part - Mechanical model
l = 1.03*1000   # mm
m = 77.7        # kg
ddx_num, ddphi_num = solv.symbolic_solver( m, l ) 

for key in sim_ids:

    # SLED
    sing_rep_res = krc_data_reader.get_single_replication_results(key)
    data = sing_rep_res['10SLEDFRMI00ACX0'] # SLED
    filtered_data = filter.filterSAEJ211(data, 6, 0.001)
    sled_acceleration_data = np.array(filtered_data) # mm/s^2
    # PELVIS
    filtered_pelv_acx_data = filter.filterSAEJ211( sing_rep_res['11PELV0000H3ACX0'], 6, 0.001 ) # PELVIS X direction
    pelvis_acceleration_simu_x = np.array( filtered_pelv_acx_data ) # mm/s^2
    filtered_pelv_acy_data = filter.filterSAEJ211( sing_rep_res['11PELV0000H3ACY0'], 6, 0.001 ) # PELVIS Y direction
    pelvis_acceleration_simu_y = np.array( filtered_pelv_acy_data ) # mm/s^2
    # CHST
    filtered_chst_acx_data = filter.filterSAEJ211( sing_rep_res['11CHST0000H3ACX0'], 6, 0.001 )
    chest_acceleration_simu_x = np.array( filtered_chst_acx_data ) # mm/s^2
    filtered_chst_acy_data = filter.filterSAEJ211( sing_rep_res['11CHST0000H3ACY0'], 6, 0.001 )
    chest_acceleration_simu_y = np.array( filtered_chst_acy_data ) # mm/s^2
    # HEAD
    filtered_head_acx_data = filter.filterSAEJ211( sing_rep_res['11HEAD0000H3ACX0'], 6, 0.001 )
    head_acceleration_simu_x = np.array( filtered_head_acx_data ) # mm/s^2
    filtered_head_acy_data = filter.filterSAEJ211( sing_rep_res['11HEAD0000H3ACY0'], 6, 0.001 )
    head_acceleration_simu_y = np.array( filtered_head_acy_data ) # mm/s^2

    # Sled displacement
    t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement(simulation_time_vals, sled_acceleration_data)
    gain = -1 * sled_d   # mm, minus sign: sled is pulled backwards, so it is moving in the negative direction in the global CS, but in the datafile, it was logged unsigned.
    
    ### FIT MODEL PARAMETERS TO CHANNEL DATA ###

    channels_to_use = { "pelvis" : 1, "chest" : 0, "head" : 0 } # Values: 0 for exclude 1 for include
    y0 = np.array ([0,0,0,0])
    k_initial = 3e5   # initial guess
    k_t_initial = 1e5   # initial guess

    ### !!!! Figyelem: ide még lehetne tenni egy grid_search funkciót, de a futtatási idő rövidítésének érdekében ezt most kihagyom


    # az alábbi pár sor jelen esetben lehetséges, hogy kihagyható
    # x_computed, dx_computed, phi_computed, dphi_computed, t_computed = solv.num_solver(ddx_num, ddphi_num, gain, t_values, y0, k_initial, k_t_initial) # displacements in mm, velocities in mm/s
    # acceleration_computed, angular_acceleration_computed = solv.acceleration_substituted(ddx_num, ddphi_num, gain, t_computed, y0, k_initial, k_t_initial) #mm/s^2
    # acceleration_computed = -acceleration_computed  # mm/s^2
    
    # ide még kell egy rész, ami az acceleration_computed nevű változót bepakolja a dictionarybe menő array 2 oszlopába
    #lmfit for optimizing k
    params=Parameters()
    params.add( 'k', min=0, max = None, value=k_initial , vary = True ) # N/mm
    params.add( 'k_t', min=0, max = None, value=k_t_initial, vary = True ) # Nmm/rad

    
    
    fitter = Minimizer(obj_function, params, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use))
    result = fitter.minimize(method = 'least_squares')
    # optimal_params = np.array([result.params[param].value for param in result.params])
    k_opt = result.params['k']
    k_t_opt = result.params['k_t']
    result.params.pretty_print()

    a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_opt, k_t_opt, l )

    

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

# Save 
folder = 'saved_data'
data_saver.save(data_dict_p, folder, 'EU2_11PELV0000H3ACX0s')
data_saver.save(data_dict_ch, folder, 'EU2_11CHST0000H3ACX0')
data_saver.save(data_dict_p, folder, 'EU2_11HEAD0000H3ACX0')
