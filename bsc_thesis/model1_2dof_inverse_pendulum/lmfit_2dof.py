import lmfit
from scipy.interpolate import interp1d
import model1_2dof_inverse_pendulum.equation_solver_2DoF as solv
import numpy as np


def data_interpolated( data, time ):

    data_interpolated = interp1d( time, data )
    
    return data_interpolated


def combined_obj_function( params, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k = params['k']
    k_t = params['k_t']

    a_pelvis_x, a_pelvis_y, a_chest_x, a_chest_y, a_head_x, a_head_y = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, sled_disp, simulation_time_vals, y0, k, k_t, l )

    return channels_to_use["pelvis"] * ( data_pelv - a_pelvis_x )**2  + channels_to_use["chest"] * ( data_chest - a_chest_x )**2 + channels_to_use["head"] * ( data_head - a_head_x )**2
    
def pelvis_obj_function(params, ddx_num, ddphi_num, sled_disp, simulation_time_vals, y0, k_values, k_t_values, pelvis_acceleration_data):
    k_values = params['k']
    k_t_values = params['k_t']
    y0 = np.array( [0,0,0,0] )

    acceleration_computed, angular_acceleration_computed = \
        solv.acceleration_substituted( ddx_num, ddphi_num, sled_disp, simulation_time_vals, y0, k_values, k_t_values )
    acceleration_computed = -acceleration_computed

    return ( pelvis_acceleration_data - acceleration_computed )**2

def grid_search_obj_combined( params_init_vals, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k_init = params_init_vals["k_init"]
    k_t_init = params_init_vals["k_t_init"]

    params = lmfit.Parameters()
    params.add('k', min=0, max = 1e7, value=k_init, vary = True ) # N/mm
    params.add('k_t', min=0, max = 1e7, value=k_t_init, vary = True ) # Nmm/rad 

    results = lmfit.minimize( combined_obj_function, params, "least_squares", args=( ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ) )

    return sum( results.residual )

def optuna_obj_combined( trial, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k_init = trial.suggest_float( "k_init", low=0.0, high=10.0e7 )
    k_t_init = trial.suggest_float( "k_t_init", low=0.0, high=10.0e7 )

    params = lmfit.Parameters()
    params.add('k', min=0, max = 1e7, value=k_init, vary = True ) # N/mm
    params.add('k_t', min=0, max = 1e7, value=k_t_init, vary = True ) # Nmm/rad 

    results = lmfit.minimize( combined_obj_function, params, "least_squares", args=( ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ) )

    return sum( results.residual )

