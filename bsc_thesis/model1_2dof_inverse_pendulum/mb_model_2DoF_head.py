# mb_model_2DoF.py

"""
Solving equation of motion and generating data for training

Important notice: lmfit for k and k_t needs to be implemented

"""

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
# print( channels_in_data )

# Preparing lmfit
pelvis_obj_func = lmfit_2dof.pelvis_obj_function

# Creating dictionary
sim_ids = krc_data_reader.get_available_replications()
data_dict = dict()

# Symbolic part - Mechanical model
ddx_num, ddphi_num = solv.symbolic_solver(77.7, 1.03)   # data can be modified


for key in sim_ids:

    # SLED
    sing_rep_res = krc_data_reader.get_single_replication_results(key)
    # plt.plot( sing_rep_res['10SLEDFRMI00ACX0'] )
    # Filter
    data = sing_rep_res['10SLEDFRMI00ACX0'] # SLED
    filtered_data = filter.filterSAEJ211(data, 6, 0.001)
    sled_acceleration_data = np.array(filtered_data) # mm/s^2
    # plt.plot(filtered_data)
    # ide még kell egy rész, ami a sled_acceleration_data nevű változót bepakolja a dictionarybe menő array 1. oszlopába

    # Gain: displacement
    sled_acceleration_data = sled_acceleration_data# mm/s^2
    # time = np.linspace(0, 0.00014*10**6, 1400) # usec
    # Sled displacement
    t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement(simulation_time_vals, sled_acceleration_data)
    gain = sled_d   # mm
    # Numerical solution
    y0 = np.array ([0,0,0,0])
    k_initial = 1e5   # initial guess
    k_t_initial = 3e5   # initial guess
    x_computed, dx_computed, phi_computed, dphi_computed, t_computed = solv.num_solver(ddx_num, ddphi_num, gain, t_values, y0, k_initial, k_t_initial) # displacements in mm, velocities in mm/s
    acceleration_computed, angular_acceleration_computed = solv.acceleration_substituted(ddx_num, ddphi_num, gain, t_computed, y0, k_initial, k_t_initial) #mm/s^2
    acceleration_computed = -acceleration_computed  # mm/s^2
    
    # ide még kell egy rész, ami az acceleration_computed nevű változót bepakolja a dictionarybe menő array 2 oszlopába

    # Simulation data
    simu_data = sing_rep_res['11PELV0000H3ACX0']    # PELVIS X direction
    filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
    pelvis_acceleration_simu = np.array(filtered_simu_data)    # mm/s
    # acceleration_measured = np.array(filtered_simu_data)    # 3. oszlop

    #lmfit for optimizing k
    params=Parameters()
    # set up the initial parameter values
    params.add('k', min=0, max = None, value=1e5, vary = True)    # N/mm
    params.add('k_t', min=0, max = None, value=3e5, vary = True) # Nmm/rad
    fitter = Minimizer(pelvis_obj_func, params, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_initial, k_t_initial, pelvis_acceleration_simu))
    result = fitter.minimize()
    # optimal_params = np.array([result.params[param].value for param in result.params])
    k_opt = result.params['k']
    k_t_opt = result.params['k_t']
    # result.params.pretty_print()
    pelvis_acceleration_fitted, _ = \
    solv.acceleration_substituted( ddx_num, ddphi_num, gain, t_computed, y0, k_opt, k_t_opt )
    pelvis_acceleration_fitted = -pelvis_acceleration_fitted


    # Difference
    difference = solv.difference_calc(pelvis_acceleration_simu, pelvis_acceleration_fitted) # 4. oszlop
    tmp = [sled_acceleration_data, pelvis_acceleration_fitted, pelvis_acceleration_simu, difference]
    data_dict[key] = tmp  # ennek kell beadni azt a tmp numpy arrayt amiben a 4x1400 adat van

print(data_dict)

# Save 
folder = 'saved_data'
data_saver.save(data_dict, folder, 'EU2_11PELV0000H3ACX0s')    # needs to be modified if new channels are needed


