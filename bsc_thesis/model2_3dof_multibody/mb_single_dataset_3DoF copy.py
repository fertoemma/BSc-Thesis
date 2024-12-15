# Import official modules
import os
import sys
import numpy as np
import sympy as sp
from lmfit import Parameters, Minimizer
import lmfit
import matplotlib.pyplot as plt

# Import custom modules
osd_path = os.path.dirname( os.path.dirname( __file__ ) )  # szülőkönyvtár szülőkönyvtára
sys.path.append( osd_path )

import lmfit_3dof
import model2_3dof_multibody.equation_solver_3DoF as solv
import function_files.krc_reader as krc_reader
import function_files.filterSAEJ211 as filter




### PRINTING SETTINGS ###
sp.init_printing()


### PARSING, READING AND CLEANING THE SIMULATION CHANNEL DATABASE ###

base_path = os.path.join( osd_path, 'Dataset' )
path_EU = os.path.join( base_path, 'EU2' )
path_US = os.path.join( base_path, 'US' )
# sim_repl_lib_paths = [ path_US  ]
sim_repl_lib_paths = [ path_EU  ]

krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )

krc_data_reader.load_replications_results( moving_average_win_len=10 )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed()


### READING RELEVANT CHANNELS FOR A SINGLE SIMULATION ###

simulation_time_vals = krc_data_reader.get_time_values()
simulation_channels_data = krc_data_reader.get_single_replication_results(('EU2', 'SO006'))

# SLED
filtered_sled_acx_data = filter.filterSAEJ211( simulation_channels_data['10SLEDFRMI00ACX0'], 6, 0.001 ) # SLED X direction
sled_acceleration_data = np.array( filtered_sled_acx_data ) # mm/s^2
# PELVIS
filtered_pelv_acx_data = filter.filterSAEJ211( simulation_channels_data['11PELV0000H3ACX0'], 6, 0.001 ) # PELVIS X direction
pelvis_acceleration_simu_x = np.array( filtered_pelv_acx_data ) # mm/s^2
filtered_pelv_acy_data = filter.filterSAEJ211( simulation_channels_data['11PELV0000H3ACY0'], 6, 0.001 ) # PELVIS Y direction
pelvis_acceleration_simu_y = np.array( filtered_pelv_acy_data ) # mm/s^2
# CHST
filtered_chst_acx_data = filter.filterSAEJ211( simulation_channels_data['11CHST0000H3ACX0'], 6, 0.001 )
chest_acceleration_simu_x = np.array( filtered_chst_acx_data ) # mm/s^2
filtered_chst_acy_data = filter.filterSAEJ211( simulation_channels_data['11CHST0000H3ACY0'], 6, 0.001 )
chest_acceleration_simu_y = np.array( filtered_chst_acy_data ) # mm/s^2
# HEAD
filtered_head_acx_data = filter.filterSAEJ211( simulation_channels_data['11HEAD0000H3ACX0'], 6, 0.001 )
head_acceleration_simu_x = np.array( filtered_head_acx_data ) # mm/s^2
filtered_head_acy_data = filter.filterSAEJ211( simulation_channels_data['11HEAD0000H3ACY0'], 6, 0.001 )
head_acceleration_simu_y = np.array( filtered_head_acy_data ) # mm/s^2

### SYMBOLICALLY DERIVING THE MODELS EOMs AND DETERMINING THE SLED DISPLACEMENT ###

# Symbolic part - Mechanical model 
l = 1.03*1000-180   # mm
R = 9/100*1000           # mm
m_1 = 34        # kg
m_2 = 4         # kg
ddx_num, ddphi_num, ddtheta_num = solv.lagrangian_fast2(m_1, m_2, l, R)   # data can be modified

# Numerically integrating the SLED displacement from SLED acceleration
t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement( simulation_time_vals, sled_acceleration_data )
gain = -1 * sled_d   # mm, minus sign: sled is pulled backwards, so it is moving in the negative direction in the global CS, but in the datafile, it was logged unsigned.

### FIT MODEL PARAMETERS TO CHANNEL DATA ###

channels_to_use = { "pelvis" : 1, "chest" : 0, "head" : 0 } # Values: 0 for exclude 1 for include
y0 = np.array( [ 0, 0, 0, 0, 0, 0 ] )

params_grid = Parameters()
# set up the initial parameter values
params_grid.add('k_1_init', min=0, max = 1e6, value=0, vary = True, brute_step=5e5 ) # N/mm
params_grid.add('k_2_init', min=0, max = 1e6, value=0, vary = True, brute_step=5e5 ) # N/mm
params_grid.add('kt_1_init', min=0, max = 1e6, value=0, vary = True, brute_step=5e5 ) # Nmm/rad
params_grid.add('kt_2_init', min=0, max = 1e6, value=0, vary = True, brute_step=5e5 ) # Nmm/rad

results_grid = lmfit.minimize( lmfit_3dof.grid_search_obj_combined,
                               params_grid,
                               method = "brute",
                               args=( ddx_num, ddphi_num, ddtheta_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, R, l, channels_to_use ) )

print("Grid results:")
results_grid.params.pretty_print()

k_1_init_opt = results_grid.params['k_1_init']
k_2_init_opt = results_grid.params['k_2_init']
kt_1_init_opt = results_grid.params['kt_1_init']
kt_2_init_opt = results_grid.params['kt_2_init']

params = Parameters()
# set up the initial parameter values
params.add( 'k_1', min=0, max = None, value=k_1_init_opt , vary = True ) # N/mm
params.add( 'kt_1', min=0, max = None, value=kt_1_init_opt, vary = True ) # Nmm/rad
params.add( 'k_2', min=0, max = None, value=k_2_init_opt , vary = True ) # N/mm
params.add( 'kt_2', min=0, max = None, value=kt_2_init_opt, vary = True ) # Nmm/rad

fitter = Minimizer( lmfit_3dof.combined_obj_function, params,
                     fcn_args=( ddx_num, ddphi_num, ddtheta_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, R, l, channels_to_use ) )
result = fitter.minimize( method='least_squares' )

k_1_opt = result.params['k_1']
kt_1_opt = result.params['kt_1']
k_2_opt = result.params['k_2']
kt_2_opt = result.params['kt_2']
print("Optimized results:")
result.params.pretty_print()

# Computing acceleration values using combined fitting
a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, ddtheta_num, gain, simulation_time_vals, y0, k_1_opt, k_2_opt, kt_1_opt, kt_2_opt, R, l )



plt.figure()
plt.plot( simulation_time_vals, pelvis_acceleration_simu_x, label = 'simu')
plt.plot( simulation_time_vals, a_pelvis_x_combfit, label ='pelvis fitted')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s^2]')
plt.title("Comparison of simulated and computed results CHEST")
plt.show()

plt.figure()
plt.plot( simulation_time_vals, chest_acceleration_simu_x, label = 'simu')
plt.plot( simulation_time_vals, a_chest_x_combfit, label ='chest fitted')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s^2]')
plt.title("Comparison of simulated and computed results CHEST")
plt.show()

plt.figure()
plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
plt.plot( simulation_time_vals, a_head_x_combfit, label ='fitted')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s^2]')
plt.title("Comparison of simulated and computed results HEAD")
plt.show()
