# Import official modules
import os
import sys
import numpy as np
import sympy as sp
from lmfit import Parameters, Minimizer
import lmfit
import matplotlib.pyplot as plt
import errno
import optuna
import pandas as pd
# Import custom modules
osd_path = os.path.dirname(os.path.dirname(__file__))  # szülőkönyvtár szülőkönyvtára
sys.path.append(osd_path)

import model3_2dof_2spring.lmfit_2dof_modified as lmfit_2dof
import model3_2dof_2spring.equation_solver_2DoF as solv
import function_files.krc_reader as krc_reader
import function_files.filterSAEJ211 as filter

### PRINTING SETTINGS ###
sp.init_printing()

### PARSING, READING AND CLEANING THE SIMULATION CHANNEL DATABASE ###

base_path = os.path.join(osd_path, 'Dataset')
path_EU = os.path.join(base_path, 'EU2')
path_US = os.path.join(base_path, 'US')
# sim_repl_lib_paths = [ path_US  ]
sim_repl_lib_paths = [path_EU]

krc_data_reader = krc_reader.KeyResultCurveDataReader(sim_repl_lib_paths)

krc_data_reader.load_replications_results(moving_average_win_len=10)
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications(['11CHST0000H3DSX0'])
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications([('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095')])  # ('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed()

### READING RELEVANT CHANNELS FOR A SINGLE SIMULATION ###

simulation_time_vals = krc_data_reader.get_time_values()
simulation_channels_data = krc_data_reader.get_single_replication_results(('EU2', 'SO006'))

# SLED
filtered_sled_acx_data = filter.filterSAEJ211(simulation_channels_data['10SLEDFRMI00ACX0'], 6, 0.001)  # SLED X direction
sled_acceleration_data = np.array(filtered_sled_acx_data)  # mm/s^2
# PELVIS
filtered_pelv_acx_data = filter.filterSAEJ211(simulation_channels_data['11PELV0000H3ACX0'], 6, 0.001)  # PELVIS X direction
pelvis_acceleration_simu_x = np.array(filtered_pelv_acx_data)  # mm/s^2
filtered_pelv_acy_data = filter.filterSAEJ211(simulation_channels_data['11PELV0000H3ACY0'], 6, 0.001)  # PELVIS Y direction
pelvis_acceleration_simu_y = np.array(filtered_pelv_acy_data)  # mm/s^2
# CHST
filtered_chst_acx_data = filter.filterSAEJ211(simulation_channels_data['11CHST0000H3ACX0'], 6, 0.001)
chest_acceleration_simu_x = np.array(filtered_chst_acx_data)  # mm/s^2
filtered_chst_acy_data = filter.filterSAEJ211(simulation_channels_data['11CHST0000H3ACY0'], 6, 0.001)
chest_acceleration_simu_y = np.array(filtered_chst_acy_data)  # mm/s^2
# HEAD
filtered_head_acx_data = filter.filterSAEJ211(simulation_channels_data['11HEAD0000H3ACX0'], 6, 0.001)
head_acceleration_simu_x = np.array(filtered_head_acx_data)  # mm/s^2
filtered_head_acy_data = filter.filterSAEJ211(simulation_channels_data['11HEAD0000H3ACY0'], 6, 0.001)
head_acceleration_simu_y = np.array(filtered_head_acy_data)  # mm/s^2

### SYMBOLICALLY DERIVING THE MODELS EOMs AND DETERMINING THE SLED DISPLACEMENT ###

# Symbolic part - Mechanical model
l = 1.03*1000   # mm
m = 77.7        # kg
ddx_num, ddphi_num = solv.symbolic_solver( m, l)   # data can be modified

# Numerically integrating the SLED displacement from SLED acceleration
t_values, sled_v, sled_d = solv.sled_acceleration_to_displacement(simulation_time_vals, sled_acceleration_data)
gain = -1 * sled_d  # mm, minus sign: sled is pulled backwards, so it is moving in the negative direction in the global CS, but in the datafile, it was logged unsigned.

### FIT MODEL PARAMETERS TO CHANNEL DATA ###

channels_to_use = {"pelvis": 1, "chest": 1, "head": 1}  # Values: 0 for exclude 1 for include
y0 = np.array( [ 0, 0, 0, 0 ] )

# Define the main directory for Optuna studies
optuna_2dof_opt_main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_2dof_2_k_inits_pelv_chst_head_obj")

# Define the study name
study_name = "3dof_k_inits"

# Define the storage name (path to the SQLite database)
storage_name = os.path.join(f"sqlite:///{optuna_2dof_opt_main_dir}", f"{study_name}.db")

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_name)

best_init_vals = study.best_trial.params

print(best_init_vals)

k_1_init_opt = best_init_vals['k_1_init']
kt_1_init_opt = best_init_vals['kt_1_init']
k_2_init_opt = best_init_vals['k_2_init']


# Local optimisation with lmfit to locally refine the parameter values, using the best inital parameter values from the global optimisation.
params = Parameters()
# set up the initial parameter values
params.add('k_1', min=0, max=None, value=k_1_init_opt, vary=True)  # N/mm
params.add('kt_1', min=0, max=None, value=kt_1_init_opt, vary=True)  # Nmm/rad
params.add('k_2', min=0, max=None, value=k_2_init_opt, vary=True)  # N/mm


fitter = Minimizer(lmfit_2dof.combined_obj_function, params,
                   fcn_args=(ddx_num, ddphi_num, gain, simulation_time_vals, pelvis_acceleration_simu_x, chest_acceleration_simu_x, head_acceleration_simu_x, y0, l, channels_to_use))
result = fitter.minimize(method='least_squares')

k_1_opt = result.params['k_1']
kt_1_opt = result.params['kt_1']
k_2_opt = result.params['k_2']
result.params.pretty_print()

a_pelvis_x_combfit, a_pelvis_y_combfit, a_chest_x_combfit, a_chest_y_combfit, a_head_x_combfit, a_head_y_combfit = \
    solv.calculate_acceleration_components_for_all_body_parts(ddx_num, ddphi_num, gain, simulation_time_vals, y0, k_1_opt, k_2_opt, kt_1_opt, l)

# Plotting the resutls

# Plotting with LaTeX font
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

plt.figure()
plt.plot(simulation_time_vals, pelvis_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_pelvis_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)  
plt.title(f"Comparison of simulation and fitted E.o.M results\nPELVIS", fontsize=12)
plt.show()

plt.figure()
plt.plot(simulation_time_vals, chest_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_chest_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)  
plt.title(f"Comparison of simulation and fitted E.o.M results\nCHEST", fontsize=12)
plt.show()

plt.figure()
plt.plot(simulation_time_vals, head_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_head_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)  
plt.title(f"Comparison of simulation and fitted E.o.M results\nHEAD", fontsize=12)
plt.show()

# Save the plots as PNG and SVG
output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
os.makedirs(output_dir, exist_ok=True)

# Save pelvis plot
plt.figure()
plt.plot(simulation_time_vals, pelvis_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_pelvis_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)
plt.title(f"Comparison of simulation and fitted E.o.M results\nPELVIS", fontsize=12)
plt.savefig(os.path.join(output_dir, 'pelvis_plot.png'))
plt.savefig(os.path.join(output_dir, 'pelvis_plot.svg'))
plt.close()

# Save chest plot
plt.figure()
plt.plot(simulation_time_vals, chest_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_chest_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)
plt.title(f"Comparison of simulation and fitted E.o.M results\nCHEST", fontsize=12)
plt.savefig(os.path.join(output_dir, 'chest_plot.png'))
plt.savefig(os.path.join(output_dir, 'chest_plot.svg'))
plt.close()

# Save head plot
plt.figure()
plt.plot(simulation_time_vals, head_acceleration_simu_x, label="simulation", color='#86a542')
plt.plot(simulation_time_vals, a_head_x_combfit, label="fitted", color='#6881d8')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [mm/s$^2$]', fontsize=9)
plt.title(f"Comparison of simulation and fitted E.o.M results\nHEAD", fontsize=12)
plt.savefig(os.path.join(output_dir, 'head_plot.png'))
plt.savefig(os.path.join(output_dir, 'head_plot.svg'))
plt.close()

