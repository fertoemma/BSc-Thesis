import lmfit
from scipy.interpolate import interp1d
import model3_2dof_2spring.equation_solver_2DoF as solv
import numpy as np


def data_interpolated( data, time ):

    data_interpolated = interp1d( time, data )
    
    return data_interpolated


def combined_obj_function( params, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k_1 = params['k_1']
    k_2 = params['k_2']
    kt_1 = params['kt_1']

    a_pelvis_x, a_pelvis_y, a_chest_x, a_chest_y, a_head_x, a_head_y = \
        solv.calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, sled_disp, simulation_time_vals, y0, k_1, k_2, kt_1, l )

    return channels_to_use["pelvis"] * ( data_pelv - a_pelvis_x )**2  + channels_to_use["chest"] * ( data_chest - a_chest_x )**2 + channels_to_use["head"] * ( data_head - a_head_x )**2
    

def grid_search_obj_combined( params_init_vals, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k_1_init = params_init_vals["k_1_init"]
    k_2_init = params_init_vals["k_2_init"]
    kt_1_init = params_init_vals["kt_1_init"]

    params = lmfit.Parameters()
    params.add('k_1', min=0, max = 1e7, value=k_1_init, vary = True ) # N/mm
    params.add('k_2', min=0, max = 1e7, value=k_2_init, vary = True ) # N/mm
    params.add('kt_1', min=0, max = 1e7, value=kt_1_init, vary = True ) # Nmm/rad 

    results = lmfit.minimize( combined_obj_function, params, "least_squares", args=( ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ) )

    return sum( results.residual )

def optuna_obj_combined( trial, ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ):

    k_1_init = trial.suggest_float( "k_1_init", low=0.0, high=10.0e7 )
    k_2_init = trial.suggest_float( "k_2_init", low=0.0, high=10.0e7 )
    kt_1_init = trial.suggest_float( "kt_1_init", low=0.0, high=10.0e7 )

    params = lmfit.Parameters()
    params.add('k_1', min=0, max = 1e7, value=k_1_init, vary = True ) # N/mm
    params.add('k_2', min=0, max = 1e7, value=k_2_init, vary = True ) # N/mm
    params.add('kt_1', min=0, max = 1e7, value=kt_1_init, vary = True ) # Nmm/rad 

    results = lmfit.minimize( combined_obj_function, params, "least_squares", args=( ddx_num, ddphi_num, sled_disp, simulation_time_vals, data_pelv, data_chest, data_head, y0, l, channels_to_use ) )

    return sum( results.residual )