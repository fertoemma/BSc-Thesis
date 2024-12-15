# Dataprocessing for showing the differences between eu2 and us and the effect of filtering on the data


# Import official modules
import os
import sys
import numpy as np
import sympy as sp
from lmfit import Parameters, Minimizer
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

### EU2 data 
# sim_repl_lib_paths = [ path_US  ]
sim_repl_lib_paths = [ path_EU  ]

krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )

krc_data_reader.load_replications_results( moving_average_win_len=10 )   # needs to be set back to 10 after plotting results
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed()



### READING RELEVANT CHANNELS FOR A SINGLE SIMULATION ###

# Chosen dataset and simulation settings


simulation_time_vals = krc_data_reader.get_time_values()
eu_simulation_time_vals = krc_data_reader.get_time_values()
simulation_channels_data = krc_data_reader.get_single_replication_results(('EU2', 'SO077'))

# SLED
eu_filtered_sled_acx_data = filter.filterSAEJ211( simulation_channels_data['10SLEDFRMI00ACX0'], 6, 0.001 ) # SLED X direction
sled_acceleration_data = np.array( eu_filtered_sled_acx_data ) # mm/s^2
eu_unflitered_sled_acx = simulation_channels_data['10SLEDFRMI00ACX0']
# Plotting the filtered and unfiltered data for sled acceleration in X direction
output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
os.makedirs(output_dir, exist_ok=True)
# Plotting with LaTeX font
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)

# # Plot pelvis comparison
# plt.figure()
# plt.plot(simulation_time_vals, eu_unflitered_sled_acx, label="simulation", color='#86a542')
# plt.plot(simulation_time_vals, eu_filtered_sled_acx_data, label="grid fit", color='#86a542')
# # plt.plot(simulation_time_vals, a_pelvis_x_combfit_optuna, label="optuna fit", color='#b84c7d', linestyle='--')
# plt.legend()
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [mm/s$^2$]', fontsize=10)
# plt.title(f"Comparison of fitted and simulation results\nPELVIS", fontsize=12)
# plt.savefig(os.path.join(output_dir, 'pelvis_comparison_plot.png'))
# plt.savefig(os.path.join(output_dir, 'pelvis_comparison_plot.svg'))
# plt.close()


# PELVIS
eu_unfiltered_pelvis_acx = simulation_channels_data['11PELV0000H3ACX0']
eu_filtered_pelv_acx_data = filter.filterSAEJ211( simulation_channels_data['11PELV0000H3ACX0'], 6, 0.001 ) # PELVIS X direction

# Plotting the filtered and unfiltered data for pelvis acceleration in X direction
# plt.figure(figsize=(10, 6))
# plt.plot(simulation_time_vals, unfiltered_pelvis, label='Unfiltered Pelvis Acceleration X', linestyle='--')
# plt.plot(simulation_time_vals, filtered_pelv_acx_data, label='Filtered Pelvis Acceleration X')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (mm/s^2)')
# plt.title('Pelvis Acceleration in X Direction')
# plt.legend()
# plt.grid(True)
# plt.show()
pelvis_acceleration_simu_x = np.array( eu_filtered_pelv_acx_data ) # mm/s^2
filtered_pelv_acy_data = filter.filterSAEJ211( simulation_channels_data['11PELV0000H3ACY0'], 6, 0.001 ) # PELVIS Y direction
pelvis_acceleration_simu_y = np.array( filtered_pelv_acy_data ) # mm/s^2
# CHST
eu_unflitered_chest_acx = simulation_channels_data['11CHST0000H3ACX0']
eu_filtered_chst_acx_data = filter.filterSAEJ211( simulation_channels_data['11CHST0000H3ACX0'], 6, 0.001 )
chest_acceleration_simu_x = np.array( eu_filtered_chst_acx_data ) # mm/s^2
filtered_chst_acy_data = filter.filterSAEJ211( simulation_channels_data['11CHST0000H3ACY0'], 6, 0.001 )
chest_acceleration_simu_y = np.array( filtered_chst_acy_data ) # mm/s^2
# HEAD
eu_unflitered_head_acx = simulation_channels_data['11HEAD0000H3ACX0']
eu_filtered_head_acx_data = filter.filterSAEJ211( simulation_channels_data['11HEAD0000H3ACX0'], 6, 0.001 )
head_acceleration_simu_x = np.array( eu_filtered_head_acx_data ) # mm/s^2
filtered_head_acy_data = filter.filterSAEJ211( simulation_channels_data['11HEAD0000H3ACY0'], 6, 0.001 )
head_acceleration_simu_y = np.array( filtered_head_acy_data ) # mm/s^2


### US data

sim_repl_lib_paths = [ path_US  ]
krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )

krc_data_reader.load_replications_results( moving_average_win_len=1 )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()
krc_data_reader.replications_with_not_1400_elements_removed()

simulation_time_vals = krc_data_reader.get_time_values()
us_simulation_time_vals = krc_data_reader.get_time_values()
simulation_channels_data = krc_data_reader.get_single_replication_results(('US', 'SO002'))

# SLED
us_unflitered_sled_acx = simulation_channels_data['10SLEDFRMI00ACX0']
us_flitered_sled_acx = filter.filterSAEJ211( simulation_channels_data['10SLEDFRMI00ACX0'], 6, 0.001 )
# PELVIS
us_unfiltered_pelvis_acx = simulation_channels_data['11PELV0000H3ACX0']
us_filtered_pelv_acx_data = filter.filterSAEJ211(simulation_channels_data['11PELV0000H3ACX0'], 6, 0.001)
# CHST
us_unflitered_chest_acx = simulation_channels_data['11CHST0000H3ACX0']
us_filtered_chst_acx_data = filter.filterSAEJ211( simulation_channels_data['11CHST0000H3ACX0'], 6, 0.001 )
# HEAD
us_unflitered_head_acx = simulation_channels_data['11HEAD0000H3ACX0']
us_filtered_head_acx_data = filter.filterSAEJ211(simulation_channels_data['11HEAD0000H3ACX0'], 6, 0.001)
# # Plotting the EU and US acceleration data for SLED, PELVIS, CHEST, and HEAD in X direction

# Function to format the y-axis labels in scientific notation
def scientific_formatter(x, pos):
    return f'{x:.1e}'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Assuming you have already created the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(10, 8))


### UNFILTERED

# # SLED
# axs[0, 0].plot(eu_simulation_time_vals, eu_unflitered_sled_acx, label='EU2 SLED ACX', color='#8650a6')
# axs[0, 0].plot(us_simulation_time_vals, us_unflitered_sled_acx, label='US SLED ACX', color='#86a542')
# axs[0, 0].set_xlabel('Time (s)', fontsize=9)
# axs[0, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[0, 0].set_title('SLED Acceleration in X Direction', fontsize=12)
# axs[0, 0].legend()
# axs[0, 0].grid(True)
# axs[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # PELVIS
# axs[0, 1].plot(eu_simulation_time_vals, eu_unfiltered_pelvis_acx, label='EU2 PELVIS ACX', color='#8650a6')
# axs[0, 1].plot(us_simulation_time_vals, us_unfiltered_pelvis_acx, label='US PELVIS ACX', color='#86a542')
# axs[0, 1].set_xlabel('Time (s)', fontsize=9)
# axs[0, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[0, 1].set_title('Pelvis Acceleration in X Direction', fontsize=12)
# axs[0, 1].legend()
# axs[0, 1].grid(True)
# axs[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # CHEST
# axs[1, 0].plot(eu_simulation_time_vals, eu_unflitered_chest_acx, label='EU2 CHEST ACX', color='#8650a6')
# axs[1, 0].plot(us_simulation_time_vals, us_unflitered_chest_acx, label='US CHEST ACX', color='#86a542')
# axs[1, 0].set_xlabel('Time (s)', fontsize=9)
# axs[1, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[1, 0].set_title('Chest Acceleration in X Direction', fontsize=12)
# axs[1, 0].legend()
# axs[1, 0].grid(True)
# axs[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # HEAD
# axs[1, 1].plot(eu_simulation_time_vals, eu_unflitered_head_acx, label='EU2 HEAD ACX', color='#8650a6')
# axs[1, 1].plot(us_simulation_time_vals, us_unflitered_head_acx, label='US HEAD ACX', color='#86a542')
# axs[1, 1].set_xlabel('Time (s)', fontsize=9)
# axs[1, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[1, 1].set_title('Head Acceleration in X Direction', fontsize=12)
# axs[1, 1].legend()
# axs[1, 1].grid(True)
# axs[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# plt.tight_layout()

# # Save the plot in both PNG and SVG formats
# fig.savefig('plot.png', format='png', dpi=300)
# fig.savefig('plot.svg', format='svg')

# plt.show()

### FILTERED

# # EU2
# # SLED
# axs[0, 0].plot(eu_simulation_time_vals, eu_unflitered_sled_acx, label='Unfiltered SLED ACX', color='#b84c3e')
# axs[0, 0].plot(eu_simulation_time_vals, sled_acceleration_data, label='Filtered SLED ACX', color='#8650a6')
# axs[0, 0].set_xlabel('Time (s)', fontsize=9)
# axs[0, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[0, 0].set_title('SLED Acceleration in X Direction (EU2)', fontsize=12)
# axs[0, 0].legend()
# axs[0, 0].grid(True)
# axs[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # PELVIS
# axs[0, 1].plot(eu_simulation_time_vals, eu_unfiltered_pelvis_acx, label='Unfiltered PELVIS ACX', color='#b84c3e')
# axs[0, 1].plot(eu_simulation_time_vals, pelvis_acceleration_simu_x, label='Filtered PELVIS ACX', color='#8650a6')
# axs[0, 1].set_xlabel('Time (s)', fontsize=9)
# axs[0, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[0, 1].set_title('Pelvis Acceleration in X Direction (EU2)', fontsize=12)
# axs[0, 1].legend()
# axs[0, 1].grid(True)
# axs[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # CHEST
# axs[1, 0].plot(eu_simulation_time_vals, eu_unflitered_chest_acx, label='Unfiltered CHEST ACX', color='#b84c3e')
# axs[1, 0].plot(eu_simulation_time_vals, chest_acceleration_simu_x, label='Filtered CHEST ACX', color='#8650a6')
# axs[1, 0].set_xlabel('Time (s)', fontsize=9)
# axs[1, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[1, 0].set_title('Chest Acceleration in X Direction (EU2)', fontsize=12)
# axs[1, 0].legend()
# axs[1, 0].grid(True)
# axs[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# # HEAD
# axs[1, 1].plot(eu_simulation_time_vals, eu_unflitered_head_acx, label='Unfiltered HEAD ACX', color='#b84c3e')
# axs[1, 1].plot(eu_simulation_time_vals, head_acceleration_simu_x, label='Filtered HEAD ACX', color='#8650a6')
# axs[1, 1].set_xlabel('Time (s)', fontsize=9)
# axs[1, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
# axs[1, 1].set_title('Head Acceleration in X Direction (EU2)', fontsize=12)
# axs[1, 1].legend()
# axs[1, 1].grid(True)
# axs[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# plt.tight_layout()

# # Save the plot in both PNG and SVG formats
# fig.savefig('eu2_filter_comparison.png', format='png', dpi=300)
# fig.savefig('eu2_filter_comparison.svg', format='svg')

# plt.show()


# US
# SLED
axs[0, 0].plot(us_simulation_time_vals, us_unflitered_sled_acx, label='Unfiltered SLED ACX', color='#b84c3e')
axs[0, 0].plot(us_simulation_time_vals, us_flitered_sled_acx, label='Filtered SLED ACX', color='#86a542')
axs[0, 0].set_xlabel('Time (s)', fontsize=9)
axs[0, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
axs[0, 0].set_title('SLED Acceleration in X Direction (US)', fontsize=12)
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# PELVIS
axs[0, 1].plot(us_simulation_time_vals, us_unfiltered_pelvis_acx, label='Unfiltered PELVIS ACX', color='#b84c3e')
axs[0, 1].plot(us_simulation_time_vals, us_filtered_pelv_acx_data, label='Filtered PELVIS ACX', color='#86a542')
axs[0, 1].set_xlabel('Time (s)', fontsize=9)
axs[0, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
axs[0, 1].set_title('Pelvis Acceleration in X Direction (US)', fontsize=12)
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# CHEST
axs[1, 0].plot(us_simulation_time_vals, us_unflitered_chest_acx, label='Unfiltered CHEST ACX', color='#b84c3e')
axs[1, 0].plot(us_simulation_time_vals, us_filtered_chst_acx_data, label='Filtered CHEST ACX', color='#86a542')
axs[1, 0].set_xlabel('Time (s)', fontsize=9)
axs[1, 0].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
axs[1, 0].set_title('Chest Acceleration in X Direction (US)', fontsize=12)
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

# HEAD
axs[1, 1].plot(us_simulation_time_vals, us_unflitered_head_acx, label='Unfiltered HEAD ACX', color='#b84c3e')
axs[1, 1].plot(us_simulation_time_vals, us_filtered_head_acx_data, label='Filtered HEAD ACX', color='#86a542')
axs[1, 1].set_xlabel('Time (s)', fontsize=9)
axs[1, 1].set_ylabel('Acceleration (mm/s$^2$)', fontsize=9)
axs[1, 1].set_title('Head Acceleration in X Direction (US)', fontsize=12)
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

plt.tight_layout()

# Save the plot in both PNG and SVG formats
fig.savefig('us_filter_comparison.png', format='png', dpi=300)
fig.savefig('us_filter_comparison.svg', format='svg')

plt.show()

