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
# print( channels_in_data )

# Preparing lmfit
pelvis_obj_func = lmfit_2dof.pelvis_obj_function
combined_obj_func = lmfit_2dof.combined_obj_function
# Creating dictionary
sim_ids = krc_data_reader.get_available_replications()

# Simulation data

# SLED
sing_rep_res = krc_data_reader.get_single_replication_results(('EU2', 'SO006'))
# plt.plot( sing_rep_res['10SLEDFRMI00ACX0'] )
# Filter
data = sing_rep_res['10SLEDFRMI00ACX0'] # SLED
filtered_data = filter.filterSAEJ211(data, 6, 0.001)
sled_acceleration_data = np.array(filtered_data) # mm/s^2
# Pelvis
simu_data = sing_rep_res['11PELV0000H3ACX0']    # PELVIS X direction
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
pelvis_acceleration_simu_x = np.array(filtered_simu_data)    # mm/s
simu_data = sing_rep_res['11PELV0000H3ACY0']    # PELVIS X direction
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
pelvis_acceleration_simu_y = np.array(filtered_simu_data)    # mm/s
# Head 
simu_data = sing_rep_res['11HEAD0000H3ACX0']
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
head_acceleration_simu_x = np.array(filtered_simu_data)    # mm/s
simu_data = sing_rep_res['11HEAD0000H3ACY0']
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
head_acceleration_simu_y = np.array(filtered_simu_data)    # mm/s

# Chest 
simu_data = sing_rep_res['11CHST0000H3ACX0']
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
chest_acceleration_simu_x = np.array(filtered_simu_data)    # mm/s
simu_data = sing_rep_res['11CHST0000H3ACY0']
filtered_simu_data = filter.filterSAEJ211(simu_data, 6, 0.001)
chest_acceleration_simu_y = np.array(filtered_simu_data)    # mm/s

# plt.figure()
# plt.plot(head_acceleration_simu_x)
# plt.show()


# Solving the equations - numerical part

# Symbolic part - Mechanical model
l = 1.03*1000   # mm
m = 77.7        # kg
ddx_num, ddphi_num = solv.symbolic_solver(m, l)   # data can be modified

# Sled displacement
t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement(simulation_time_vals, sled_acceleration_data)
gain = -sled_d   # mm

# Numerical solution
y0 = np.array ([0,0,0,0])
k_initial = 2e+05   # initial guess
k_t_initial = 10  # initial guess
x_computed, dx_computed, phi_computed, dphi_computed, t_computed = solv.num_solver(ddx_num, ddphi_num, gain, t_values, y0, k_initial, k_t_initial) # displacements in mm, velocities in mm/s

# pelvis acceleration
acceleration_computed, angular_acceleration_computed = solv.acceleration_substituted(ddx_num, ddphi_num, gain, t_computed, y0, k_initial, k_t_initial) #mm/s^2
pelvis_acceleration_computed = acceleration_computed  # mm/s^2
a_pelvis_x = pelvis_acceleration_computed   # mm/s^2 
a_pelvis_y = np.zeros_like(a_pelvis_x)  # mm/s^2

#lmfit for pelvis
params=Parameters()
# set up the initial parameter values
params.add('k', min=0, max = None, value=4.134e+05, vary = True)    # N/mm
params.add('k_t', min=0, max = None, value=1.779, vary = True) # Nmm/rad
fitter = Minimizer(pelvis_obj_func, params, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_initial, k_t_initial, pelvis_acceleration_simu_x))
result = fitter.minimize()
# optimal_params = np.array([result.params[param].value for param in result.params])
k_opt = result.params['k']
k_t_opt = result.params['k_t']
# result.params.pretty_print()
pelvis_acceleration_fitted, _ = \
solv.acceleration_substituted( ddx_num, ddphi_num, gain, t_computed, y0, k_opt, k_t_opt )
# pelvis_acceleration_fitted = -pelvis_acceleration_fitted


# Define the output directory for the plots
output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
os.makedirs(output_dir, exist_ok=True)
# Plotting with LaTeX font
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.figure()
plt.plot( simulation_time_vals, pelvis_acceleration_simu_x, label ='simulations', color='#86a542')
# plt.plot( simulation_time_vals, pelvis_acceleration_fitted, label = 'E.o.M. solution fitted')
plt.plot( simulation_time_vals, a_pelvis_x, label = 'E.o.M. solution', color='#8650a6')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=10)
plt.title("Comparison of simulated and computed results (from initial parameter guess)\nPELVIS")
# plt.show()
# plt.savefig(os.path.join(output_dir, 'pelvis_initial_guess.png'))
plt.savefig(os.path.join(output_dir, 'pelvis_initial_guess.svg'))
# plt.close()

# head acceleration
r_ph_x = l*np.sin(phi_computed) # mm - r_ph = pelvis-head vector
r_ph_y = l*np.cos(phi_computed) # mm
print(r_ph_x.shape, r_ph_y.shape, a_pelvis_x.shape, a_pelvis_y.shape)
a_head = solv.relative_acceleration_formula(a_pelvis_x, a_pelvis_y, angular_acceleration_computed, r_ph_x, r_ph_y, dphi_computed)
print(a_head.shape)
a_head_x = a_head[0, :]  
a_head_y = a_head[1, :]  

# ???
a_head_fit = solv.relative_acceleration_formula(pelvis_acceleration_fitted, a_pelvis_y, angular_acceleration_computed, r_ph_x, r_ph_y, dphi_computed)
# ???

a_head_x_symbolic, a_head_y_symbolic = solv.symbolic_solver_head(m, l)
a_head_x_fit = a_head_fit[0, :]  
a_head_y_fit = a_head_fit[1, :]  
# plt.figure()
# plt.plot( simulation_time_vals, head_acceleration_simu_x, label ='simulations')
# plt.plot( simulation_time_vals, a_head_x_fit, label = 'E.o.M. solution fitted')
# plt.plot( simulation_time_vals, a_head_x, label = 'E.o.M. solution')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results")
# plt.show()


# chest acceleration
r_pch_x = (3/4)*l*np.sin(phi_computed) # mm - r_ph = pelvis-head vector
r_pch_y = (3/4)*l*np.cos(phi_computed) # mm
print(r_ph_x.shape, r_ph_y.shape, a_pelvis_x.shape, a_pelvis_y.shape)
a_chest = solv.relative_acceleration_formula(a_pelvis_x, a_pelvis_y, angular_acceleration_computed, r_pch_x, r_pch_y, dphi_computed)
print(a_chest.shape)
a_chest_x = a_chest[0, :]  
a_chest_y = a_chest[1, :]  

# ???
a_chest_fit = solv.relative_acceleration_formula(pelvis_acceleration_fitted, a_pelvis_y, angular_acceleration_computed, r_pch_x, r_pch_y, dphi_computed)
# ???

a_chest_x_symbolic, a_chest_y_symbolic = solv.symbolic_solver_chest(m, l)
a_chest_x_fit = a_chest_fit[0, :]  
a_chest_y = a_chest_fit[1, :]  
# plt.figure()
# plt.plot( simulation_time_vals, chest_acceleration_simu_x, label ='simulations')
# plt.plot( simulation_time_vals, a_chest_x, label = 'E.o.M. solution')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results")
# plt.show()


# Lmfit for pelvis, chest, head together
acceleration_chest_computed, acceleration_head_computed = solv.acceleration_substituted_ch(a_chest_x_symbolic, a_head_x_symbolic, gain, x_computed, dx_computed, phi_computed, dphi_computed, k_initial, k_t_initial, l) #mm/s^2
plt.figure()
plt.plot( simulation_time_vals, acceleration_head_computed, label ='head')
plt.plot( simulation_time_vals, a_head_x, label = 'E.o.M. solution')
plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
plt.legend()
plt.show()
# plt.figure()
# plt.plot( simulation_time_vals, acceleration_chest_computed, label ='chest')
# plt.plot( simulation_time_vals, a_chest_x, label = 'E.o.M. solution')
# plt.plot( simulation_time_vals, chest_acceleration_simu_x, label = 'simu')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# # plt.title("Comparison of simulated and computed results")
# plt.show()
# plt.figure()
# plt.plot( simulation_time_vals, acceleration_head_computed, label ='head')
# plt.plot( simulation_time_vals, a_head_x, label = 'E.o.M. solution')
# plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# # plt.title("Comparison of simulated and computed results")
# plt.show()

#lmfit for pelvis, head, chest together
params=Parameters()
# set up the initial parameter values
params.add('k', min=0, max = None, value= 4.134e+05 , vary = True)    # N/mm
params.add('k_t', min=0, max = None, value=1.779, vary = True) # Nmm/rad
fitter = Minimizer(combined_obj_func, params, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, x_computed, dx_computed, phi_computed, dphi_computed, a_chest_x_symbolic, a_head_x_symbolic, pelvis_acceleration_simu_x, head_acceleration_simu_x, chest_acceleration_simu_x, l))
result = fitter.minimize(method='least_squares')
# optimal_params = np.array([result.params[param].value for param in result.params])
k_opt = result.params['k']
k_t_opt = result.params['k_t']
print(f"k_opt: {type(k_opt)}, value: {getattr(k_opt, 'value', k_opt)}")
print(f"k_t_opt: {type(k_t_opt)}, value: {getattr(k_t_opt, 'value', k_t_opt)}")
k_opt_value = k_opt.value
k_t_opt_value = k_t_opt.value

result.params.pretty_print()
pelvis_acceleration_fitted, _ = \
solv.acceleration_substituted( ddx_num, ddphi_num, gain, t_computed, y0, k_opt, k_t_opt )
pelvis_acceleration_fitted = pelvis_acceleration_fitted
acceleration_chest_fitted, acceleration_head_fitted = solv.acceleration_substituted_ch(a_chest_x_symbolic, a_head_x_symbolic, gain, x_computed, dx_computed, phi_computed, dphi_computed, k_opt_value, k_t_opt_value, l) #mm/s^2


plt.figure()
plt.plot( simulation_time_vals, acceleration_head_computed, label ='head')
plt.plot( simulation_time_vals, a_head_x, label = 'E.o.M. solution')
plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
plt.plot( simulation_time_vals, acceleration_head_fitted, label = 'fitted')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results")
plt.show()

plt.figure()
plt.plot( simulation_time_vals, acceleration_chest_computed, label ='chest')
plt.plot( simulation_time_vals, a_chest_x, label = 'E.o.M. solution')
plt.plot( simulation_time_vals, chest_acceleration_simu_x, label = 'simu')
plt.plot( simulation_time_vals, acceleration_chest_fitted, label = 'fitted')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results")
plt.show()

result.params.pretty_print()
params_init_vals = lmfit.Parameters()
params_init_vals.add('k_init', min=0, max = 1e6, value=0, vary = True, brute_step=1e5 )    # N/mme
params_init_vals.add('k_t_init', min=0, max = 1e7, value=0, vary = True, brute_step=1e6 ) # Nmm/rad

fitter_2 = Minimizer(lmfit_2dof.grid_search_obj_combined, params_init_vals, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, x_computed, dx_computed, phi_computed, dphi_computed, a_chest_x_symbolic, a_head_x_symbolic, pelvis_acceleration_simu_x, head_acceleration_simu_x, chest_acceleration_simu_x, l))
# results_brute = lmfit.minimize( lmfit_2dof.grid_search_obj_combined, params_init_vals, "brute", fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, x_computed, dx_computed, phi_computed, dphi_computed, a_chest_x_symbolic, a_head_x_symbolic, pelvis_acceleration_simu_x, head_acceleration_simu_x, chest_acceleration_simu_x, l) )
results_brute = fitter_2.minimize(method='brute')
k_init_opt = results_brute.params['k_init']
k_t_init_opt = results_brute.params['k_t_init']

results_brute.params.pretty_print()



#lmfit for pelvis, head, chest together
params_2=Parameters()
# set up the initial parameter values
params_2.add('k', min=0, max = None, value= k_init_opt , vary = True)    # N/mm
params_2.add('k_t', min=0, max = None, value=k_t_init_opt, vary = True) # Nmm/rad
fitter_3 = Minimizer(combined_obj_func, params_2, fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, x_computed, dx_computed, phi_computed, dphi_computed, a_chest_x_symbolic, a_head_x_symbolic, pelvis_acceleration_simu_x, head_acceleration_simu_x, chest_acceleration_simu_x, l))
result_2 = fitter_3.minimize(method='least_squares')
# optimal_params = np.array([result.params[param].value for param in result.params])
k_opt_2 = result_2.params['k']
k_t_opt_2 = result_2.params['k_t']
result_2.params.pretty_print()
# print(f"k_opt: {type(k_opt)}, value: {getattr(k_opt, 'value', k_opt)}")
# print(f"k_t_opt: {type(k_t_opt)}, value: {getattr(k_t_opt, 'value', k_t_opt)}")
# k_opt_value = k_opt.value
# k_t_opt_value = k_t_opt.value