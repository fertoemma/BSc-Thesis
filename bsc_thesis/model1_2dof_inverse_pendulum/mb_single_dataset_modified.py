# Import official modules
import os
import sys
import numpy as np
import sympy as sp
from lmfit import Parameters, Minimizer
import lmfit
import matplotlib.pyplot as plt
import optuna
# Import custom modules
osd_path = os.path.dirname( os.path.dirname( __file__ ) )  # szülőkönyvtár szülőkönyvtára
sys.path.append( osd_path )

import lmfit_2dof
import model1_2dof_inverse_pendulum.equation_solver_2DoF as solv
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
simulation_channels_data = krc_data_reader.get_single_replication_results(('EU2', 'SO009'))

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
l = 1.03*1000   # mm
m = 77.7        # kg
ddx_num, ddphi_num = solv.symbolic_solver( m, l )   # data can be modified

# Numerically integrating the SLED displacement from SLED acceleration
t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement( simulation_time_vals, sled_acceleration_data )
gain = -1 * sled_d   # mm, minus sign: sled is pulled backwards, so it is moving in the negative direction in the global CS, but in the datafile, it was logged unsigned.
# Create a figure with subplots for LaTeX documentation
fig, axs = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

# Plot sled acceleration data
# Define the output directory for the plots
output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
os.makedirs(output_dir, exist_ok=True)
# Plotting with LaTeX font
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Create a figure with subplots for LaTeX documentation
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# Plot sled acceleration data
axs[0].plot(simulation_time_vals, sled_acceleration_data, label="sled acceleration", color='#86a542')
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Acceleration [mm/s$^2$]', fontsize=10)
axs[0].set_title("SLED Acceleration", fontsize=12)

# Plot sled displacement data
axs[1].plot(simulation_time_vals, sled_d, label="sled displacement", color='#86a542')
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Displacement [mm]', fontsize=10)
axs[1].set_title("SLED Displacement", fontsize=12)

# Save the combined plot
fig.suptitle("SLED Acceleration and Displacement", fontsize=12)
fig.savefig(os.path.join(output_dir, 'sled_acceleration_displacement_plot.png'))
fig.savefig(os.path.join(output_dir, 'sled_acceleration_displacement_plot.svg'))
plt.close()
# plt.show()


### FIT MODEL PARAMETERS TO CHANNEL DATA ###

#lmfit for pelvis, head, chest together
# params=Parameters()
# # set up the initial parameter values
# params.add('k', min=0, max = None, value=2.6e+05 , vary = True) # N/mm
# params.add('k_t', min=0, max = None, value=9.3e+05, vary = True) # Nmm/rad

# channels_to_use = { "pelvis" : 1, "chest" : 0, "head" : 0 }
# y0 = np.array( [ 0, 0, 0, 0 ] )
# fitter = Minimizer( lmfit_2dof.combined_obj_function, params,
#                      fcn_args=( ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use ) )
# result = fitter.minimize( method='least_squares' )

# k_opt = result.params['k']
# k_t_opt = result.params['k_t']
# result.params.pretty_print()

# a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
#         solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_opt, k_t_opt, l )


# plt.figure()
# plt.plot( simulation_time_vals, pelvis_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_pelvis_x_combfit, label ='pelvis fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results CHEST")
# plt.show()

# plt.figure()
# plt.plot( simulation_time_vals, chest_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_chest_x_combfit, label ='chest fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results CHEST")
# plt.show()

# plt.figure()
# plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_head_x_combfit, label ='fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results HEAD")
# plt.show()


channels_to_use = { "pelvis" : 1, "chest" : 0, "head" : 0 } # Values: 0 for exclude 1 for include
y0 = np.array( [ 0, 0, 0, 0 ] )

params_grid = Parameters()
# set up the initial parameter values

params_grid.add('k_init', min=0, max = 3e5, value=0, vary = True, brute_step=1e5 ) # N/mm
params_grid.add('k_t_init', min=0, max = 1e5, value=0, vary = True, brute_step=1e5 ) # Nmm/rad 

results_grid = lmfit.minimize( lmfit_2dof.grid_search_obj_combined,
                               params_grid,
                               method = "brute",
                               args=( ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use ) )

results_grid.params.pretty_print()

k_init_opt = results_grid.params['k_init']
k_t_init_opt = results_grid.params['k_t_init']


params = Parameters()
# set up the initial parameter values
params.add( 'k', min=0, max = None, value=k_init_opt , vary = True ) # N/mm
params.add( 'k_t', min=0, max = None, value=k_t_init_opt, vary = True ) # Nmm/rad

fitter = Minimizer( lmfit_2dof.combined_obj_function, params,
                     fcn_args=( ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use ) )
result = fitter.minimize( method='least_squares' )

k_opt_grid = result.params['k']
k_t_opt_grid = result.params['k_t']
result.params.pretty_print()

a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_opt_grid, k_t_opt_grid, l )

# Save the acceleration variables to variables with indices grid
a_pelvis_x_combfit_grid = a_pelvis_x_combfit
a_pelvis_y_combfit_grid = a_pelvis_y_combfit
a_chest_x_combfit_grid = a_chest_x_combfit
a_chest_y_combfit_grid = a_chest_y_combfit
a_head_x_combfit_grid = a_head_x_combfit
a_head_y_combfit_grid = a_head_y_combfit

# plt.figure()
# plt.plot( simulation_time_vals, pelvis_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_pelvis_x_combfit, label ='pelvis fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results PELVIS")
# plt.show()

# plt.figure()
# plt.plot( simulation_time_vals, chest_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_chest_x_combfit, label ='chest fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results CHEST")
# plt.show()

# plt.figure()
# plt.plot( simulation_time_vals, head_acceleration_simu_x, label = 'simu')
# plt.plot( simulation_time_vals, a_head_x_combfit, label ='fitted')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s^2]')
# plt.title("Comparison of simulated and computed results HEAD")
# plt.show()


# Fitting with optuna

# Define the main directory for Optuna studies
optuna_2dof_opt_main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_2dof_k_inits_pelv_obj")

# Define the study name
study_name = "2dof_k_inits"

# Define the storage name (path to the SQLite database)
storage_name = os.path.join(f"sqlite:///{optuna_2dof_opt_main_dir}", f"{study_name}.db")

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_name)

best_init_vals = study.best_trial.params

print(best_init_vals)

k_init_opt = best_init_vals['k_init']
k_t_init_opt = best_init_vals['k_t_init']



# Local optimisation with lmfit to locally refine the parameter values, using the best inital parameter values from the global optimisation.
params = Parameters()
# set up the initial parameter values
params.add('k', min=0, max=None, value=k_init_opt, vary=True)  # N/mm
params.add('k_t', min=0, max=None, value=k_t_init_opt, vary=True)  # Nmm/rad
# params.add('k_2', min=0, max=None, value=k_2_init_opt, vary=True)  # N/mm


fitter = Minimizer(lmfit_2dof.combined_obj_function, params,
                   fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use))
result = fitter.minimize(method='least_squares')

k_opt_optuna = result.params['k']
k_t_opt_optuna = result.params['k_t']

result.params.pretty_print()

a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
        solv.calculate_acceleration_components_for_all_body_parts(ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_opt_optuna, k_t_opt_optuna, l)
a_pelvis_x_combfit_optuna = a_pelvis_x_combfit
a_pelvis_y_combfit_optuna = a_pelvis_y_combfit
a_chest_x_combfit_optuna = a_chest_x_combfit
a_chest_y_combfit_optuna = a_chest_y_combfit
a_head_x_combfit_optuna = a_head_x_combfit
a_head_y_combfit_optuna = a_head_y_combfit



# Plot pelvis comparison
plt.figure()
plt.plot(simulation_time_vals, pelvis_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_pelvis_x_combfit_grid, label="grid fit", color='#50b47b')
plt.plot(simulation_time_vals, a_pelvis_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=10)
plt.title(f"Comparison of fitted and simulation results\nPELVIS", fontsize=12)
plt.savefig(os.path.join(output_dir, 'pelvis_comparison_plot.png'))
plt.savefig(os.path.join(output_dir, 'pelvis_comparison_plot.svg'))
plt.close()
# plt.show()

# Plot chest comparison
plt.figure()
plt.plot(simulation_time_vals, chest_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_chest_x_combfit_grid, label="grid fit", color='#50b47b')
plt.plot(simulation_time_vals, a_chest_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=10)
plt.title(f"Comparison of fitted and simulation results\nCHEST", fontsize=12)
plt.savefig(os.path.join(output_dir, 'chest_comparison_plot.png'))
plt.savefig(os.path.join(output_dir, 'chest_comparison_plot.svg'))
plt.close()
# plt.show()


# Plot head comparison
plt.figure()
plt.plot(simulation_time_vals, head_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_head_x_combfit_grid, label="grid fit", color='#50b47b')
plt.plot(simulation_time_vals, a_head_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=10)
plt.title(f"Comparison of fitted and simulation results\nHEAD", fontsize=12)
plt.savefig(os.path.join(output_dir, 'head_comparison_plot.png'))
plt.savefig(os.path.join(output_dir, 'head_comparison_plot.svg'))
plt.close()
# plt.show()

# Create a figure with subplots for LaTeX documentation
fig, axs = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)

# Plot pelvis comparison
axs[0].plot(simulation_time_vals, pelvis_acceleration_simu_x, label="simulation", color='#86a542')
axs[0].plot(simulation_time_vals, a_pelvis_x_combfit_grid, label="grid fit", color='#50b47b')
axs[0].plot(simulation_time_vals, a_pelvis_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Acceleration [mm/s$^2$]', fontsize=10)
axs[0].set_title("PELVIS")

# Plot chest comparison
axs[1].plot(simulation_time_vals, chest_acceleration_simu_x, label="simulation", color='#86a542')
axs[1].plot(simulation_time_vals, a_chest_x_combfit_grid, label="grid fit", color='#50b47b')
axs[1].plot(simulation_time_vals, a_chest_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Acceleration [mm/s$^2$]', fontsize=10)
axs[1].set_title("CHEST")

# Plot head comparison
axs[2].plot(simulation_time_vals, head_acceleration_simu_x, label="simulation", color='#86a542')
axs[2].plot(simulation_time_vals, a_head_x_combfit_grid, label="grid fit", color='#50b47b')
axs[2].plot(simulation_time_vals, a_head_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Acceleration [mm/s$^2$]', fontsize=10)
axs[2].set_title("HEAD")

# Save the combined plot
fig.suptitle("Comparison of fitted and simulation results", fontsize=12)
# plt.show()
fig.savefig(os.path.join(output_dir, 'combined_comparison_plot.png'))
fig.savefig(os.path.join(output_dir, 'combined_comparison_plot.svg'))
plt.close()