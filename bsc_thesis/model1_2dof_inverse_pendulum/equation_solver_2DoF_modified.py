# equation_solver.py

"""
    Solving the equation of motion numerically, computes the accelerations by 
    substituting back to the equation of motion and prepares  the data for optimizing
    with lmfit.     
"""
# Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import sympy as sp
from sympy import symbols, Function, lambdify


def sled_acceleration_to_displacement(simulation_time_vals, acceleration_data):
    # acceleration data has to be a numpy array
    acceleration_interpolated = interp1d( simulation_time_vals, acceleration_data )

    def f( t, y ):
        dx_dt = y[1]
        dv_dt = acceleration_interpolated(t)
        return [ dx_dt, dv_dt ]

    sled_sol = solve_ivp( f, ( simulation_time_vals[0], simulation_time_vals[-1] ), np.array([0, 0]), 'RK45', simulation_time_vals )
    sled_time = sled_sol.t  # s
    sled_velo = sled_sol.y[1]   # mm/s
    sled_disp = sled_sol.y[0]   # mm/s^2

    return  sled_time, sled_velo, sled_disp


# returns with a numpy function of the equation of motion
def symbolic_solver(m_data, l_data):
    # Define symbols
    t = symbols('t')
    k = symbols('k')
    k_t = symbols('k_t')
    x = Function('x')(t)
    phi = Function('phi')(t)
    r = Function('r')(t)
    dx = x.diff(t)
    ddx = dx.diff(t)
    dphi=phi.diff(t)
    ddphi=dphi.diff(t)
    # Define constants
    m, l =symbols("m l")
    # data = [(m, 77.7), (mu, 0.6), (g, 9.81), (k_t, 20), (l, 1.03)] 
    data = [(m, m_data), (l, l_data)] 
    # Kinetic and potential energy

    # A is not a permanently stationary point (the pin at the Pelvis moves). -> Kinetic energy must be written with quantities at the center of mass.
    theta= (1/12)*m*l**2 # point A # So this should change to 1/12*m*l**2
    omega = dphi
    omega_v = sp.Matrix([0,0, omega])
    r_S = sp.Matrix([l/2*sp.sin(phi), l/2*sp.cos(phi), 0])
    v_S = sp.Matrix([dx,0,0]) + omega_v.cross(r_S)

    T = (1/2)*m*v_S.dot(v_S)+(1/2)*theta*dphi**2 # And here, dx should change for v_s: dx + omega x r_s
    U = (1/2)*k*(r-x)**2+(1/2)*k_t*(phi)**2
    # Derivatives with respoect to x
    dT_dx = T.diff(x)
    dT_dxd = T.diff(dx)
    dT_dxdt = dT_dxd.diff(t)
    dU_dx = U.diff(x)
    # Derivatives with respect to phi
    dT_dphi = T.diff(phi)
    dT_dphid = T.diff(dphi)
    dT_dphidt = dT_dphid.diff(t)
    dU_dphi = U.diff(phi)
    ###
    eq1 = dT_dxdt - dT_dx + dU_dx
    eq2 = dT_dphidt - dT_dphi + dU_dphi
    # Simplifying equations
    eq1_simplified = sp.simplify(eq1)
    eq2_simplified = sp.simplify(eq2)

    eqs = sp.Matrix([eq1_simplified, eq2_simplified])
    variables = sp.Matrix([ddx, ddphi, r])
    # Solving the system
    solution = sp.solve(eqs, variables)
    # Substituting numerical values for constants
    solution_with_values = {key: value.subs(data).evalf() for key, value in solution.items()}
    # print("Solution of differential equation:", solution)
    # Conversion to numerical functions
    ddx_num_expr = solution_with_values[ddx]
    # print("ddx_num_expr:",ddx_num_expr)
    ddphi_num_expr = solution_with_values[ddphi]
    
    ddx_num = lambdify((t, x, dx, phi, dphi, r, k, k_t), ddx_num_expr, 'numpy')
    ddphi_num = lambdify((t, x, dx, phi, dphi, r, k, k_t), ddphi_num_expr, 'numpy')
    return ddx_num, ddphi_num


# gives back displacement and velocity values that can be substituted back to the equation 
def num_solver(ddx_num, ddphi_num, gain_input, time_input, y0, k, k_t):

    gain_interpolated = interp1d(time_input, gain_input)
    k = [float(k) for t in time_input]
    k = np.array(k)
    k_interpolated = interp1d(time_input, k)
    k_t = [float(k_t) for t in time_input]
    k_t = np.array(k_t)
    k_t_interpolated = interp1d(time_input, k_t)

    # Defineing the system of ODEs
    def system(t, y):
        x_val, dx_val, phi_val, dphi_val = y

        r = gain_interpolated(t)
        k = k_interpolated(t)
        k_t = k_t_interpolated(t)

        ddx_val = float(ddx_num(t, x_val, dx_val, phi_val, dphi_val, r, k, k_t))
        ddphi_val = float(ddphi_num(t, x_val, dx_val, phi_val, dphi_val, r, k, k_t))
        return [dx_val, ddx_val, dphi_val, ddphi_val]

    solution = solve_ivp( system, ( time_input[0], time_input[-1] ), y0, 'RK45', time_input )
    t_val = solution.t
    x_val = solution.y[0]
    dx_val = solution.y[1]
    phi_val = solution.y[2]
    dphi_val = solution.y[3]

    return x_val, dx_val, phi_val, dphi_val, t_val


def acceleration_substituted(ddx_num, ddphi_num, gain_input, time_input, y0,  k, k_t):
    x_values, dx_values, phi_values, dphi_values, _ = num_solver(ddx_num, ddphi_num, gain_input, time_input, y0, k, k_t)

    ddx_subst = ddx_num(time_input, x_values, dx_values, phi_values, dphi_values, gain_input, k, k_t)
    ddphi_subst = ddphi_num(time_input, x_values, dx_values, phi_values, dphi_values, gain_input, k, k_t)

    return ddx_subst, ddphi_subst


def calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, sled_disp, sim_time, y0, k, k_t, l ):

    x_computed, dx_computed, phi_computed, dphi_computed, t_computed = \
        num_solver( ddx_num, ddphi_num, sled_disp, sim_time, y0, k, k_t ) # displacements in mm, velocities in mm/s
    
    a_pelvis_x, ddphi_computed = \
        acceleration_substituted( ddx_num, ddphi_num, sled_disp, sim_time, y0, k, k_t )

    a_pelvis_y = np.zeros_like( a_pelvis_x )

    # CHEST acceleration
    r_pch_x = (3/4)*l*np.sin( phi_computed ) # mm
    r_pch_y = (3/4)*l*np.cos( phi_computed ) # mm

    a_chest = relative_acceleration_formula( a_pelvis_x, a_pelvis_y, ddphi_computed, r_pch_x, r_pch_y, dphi_computed )
    a_chest_x = a_chest[0, :]  
    a_chest_y = a_chest[1, :]

    # HEAD acceleration
    r_ph_x = l*np.sin( phi_computed ) # mm
    r_ph_y = l*np.cos( phi_computed ) # mm

    a_head = relative_acceleration_formula( a_pelvis_x, a_pelvis_y, ddphi_computed, r_ph_x, r_ph_y, dphi_computed )
    a_head_x = a_head[0, :]  
    a_head_y = a_head[1, :]

    return a_pelvis_x, a_pelvis_y, a_chest_x, a_chest_y, a_head_x, a_head_y


def relative_acceleration_formula(a_Ax, a_Ay, ddphiz, r_ABx, r_ABy, dphiz):

    a_A = np.array([a_Ax, a_Ay, np.zeros_like(a_Ax)])  # (3, 1400)
    eps = np.array([np.zeros_like(ddphiz), np.zeros_like(ddphiz), ddphiz])  # (3, 1400)
    omega = np.array([np.zeros_like(dphiz), np.zeros_like(dphiz), dphiz])  # (3, 1400)
    r_AB = np.array([r_ABx, r_ABy, np.zeros_like(r_ABx)])  # (3, 1400)


    a_B = (
        a_A +
        np.cross(eps.T, r_AB.T).T +
        np.cross(omega.T, np.cross(omega.T, r_AB.T)).T
    )

    return a_B  # (3, 1400)


def difference_calc(data_measured, data_computed):
    diff = data_measured - data_computed
    return diff