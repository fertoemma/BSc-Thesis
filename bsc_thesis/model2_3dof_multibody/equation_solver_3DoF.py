# equation_of_motion_solver.py

"""
    3DoF model with two rigid bodies
    Solver of equation of motion using the Lagrangian equation of second kind
    november, 2024
"""

# Imports 
import numpy as np
import sympy as sp
sp.init_printing()
from sympy import symbols, Function, diff, Matrix, sin, cos, lambdify
import scipy.integrate as integrate
from scipy.integrate import RK45, solve_ivp
from scipy.interpolate import interp1d



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
    
    print( sled_time == simulation_time_vals )

    return  sled_time, sled_velo, sled_disp

# returns with a numpy function of the equation of motion
def lagrangian(m_1_data, m_2_data, l_data, R_data):
    # Defining paramters and symbols
    t = symbols('t', real=True, positive=True)
    k_1 = symbols('k_1', real=True, positive=True)
    k_2 = symbols('k_2', real=True, positive=True)
    kt_1 = symbols('kt_1', real=True, positive=True)
    kt_2 = symbols('kt_2', real=True, positive=True)
    # General coordinates
    x = Function('x')(t)
    phi = Function('phi')(t)
    theta = Function('theta')(t)
    r = Function('r')(t)    # gain
    # Constans 
    m_1, m_2, l, R = symbols("m_1, m_2, l, R", real=True, positive=True)
    data = [(m_1, m_1_data), (m_2, m_2_data), (l, l_data), (R, R_data)]

    # Moment of inertia
    theta_1 = (1/12)*m_1*l**2
    theta_2 = (1/2)*m_2*R**2

    # Angular velocity
    omega_1 = phi.diff(t)
    omega_2 = phi.diff(t) + theta.diff(t)
    omega_21 = theta.diff(t)

    # Velocity
    r_S01 = Matrix([(l/2)*sin(phi), (l/2)*cos(phi), 0])
    v_S0 = Matrix([x.diff(t), 0, 0])
    omega_v1 = Matrix([0, 0, omega_1])
    v_v1 = v_S0 + omega_v1.cross(r_S01)
    # print ("Velocity of S1:", v_v1)

    r_S0C = Matrix([l*sin(phi), l*cos(phi), 0])
    omega_v2 = Matrix([0, 0, omega_2])  # not needed
    omega_v21 = Matrix([0,0,omega_21])
    v_C = v_S0 + omega_v1.cross(r_S0C)
    r_CS2 = Matrix([R*sin(theta), R*cos(theta), 0])
    v_v2 = v_C + omega_v21.cross(r_CS2)
    # print ("Velocity of S2: ", v_v2)
    # print("Magnitude of v_1 and v_2: ", v_1, "and ", v_2)

    # Displacements for k
    x_1 = x
    x_2 = x + (3/4)*l*sin(phi)

    # Angles for k_t
    phi_1 = phi
    phi_2 = theta

    # Kinetic energy
    T = (1/2)*(m_1*v_v1.dot(v_v1) + m_2*v_v2.dot(v_v2) + theta_1 * omega_1**2 + theta_2*omega_2**2)

    # Potential energy 
    U = (1/2)*(k_1*(r-x_1)**2 + k_2*(r-x_2)**2 + kt_1*phi_1**2 + kt_2*phi_2**2)

    # Derivatives
    dx = x.diff(t)
    ddx = dx.diff(t)
    dphi=phi.diff(t)
    ddphi=dphi.diff(t)
    dtheta = theta.diff(t)
    ddtheta =dtheta.diff(t)
    dT_dx = T.diff(x)
    dT_dxd = T.diff(dx)
    dT_dxdt = dT_dxd.diff(t)
    dU_dx = U.diff(x)
    dT_dphi = T.diff(phi)
    dT_dphid = T.diff(dphi)
    dT_dphidt = dT_dphid.diff(t)
    dU_dphi = U.diff(phi)
    dT_dtheta = T.diff(theta)
    dT_dthetad = T.diff(dtheta)
    dT_dthetadt = dT_dthetad.diff(t)
    dU_dtheta = U.diff(theta)
    ###
    eq1 = dT_dxdt - dT_dx + dU_dx
    eq2 = dT_dphidt - dT_dphi + dU_dphi
    eq3 = dT_dthetadt - dT_dtheta + dU_dtheta
    # Simplifying equations
    eq1_simplified = sp.simplify(eq1)
    eq2_simplified = sp.simplify(eq2)
    eq3_simplified = sp.simplify(eq3)
    eqs = sp.Matrix([eq1_simplified, eq2_simplified, eq3_simplified])
    variables = sp.Matrix([ddx, ddphi, ddtheta, r])
    # Solving the system
    solution = sp.solve(eqs, variables)
    print("Solution of differential equation:", solution)
    # Substituting numerical values for constants
    solution_with_values = {key: value.subs(data).evalf() for key, 
                            value in solution.items()}
    
    # Conversion to numerical functions
    ddx_num_expr = solution_with_values[ddx]
    ddphi_num_expr = solution_with_values[ddphi]
    ddtheta_num_expr = solution_with_values[ddtheta]
    # lambdify még nem biztos, hogy jó
    ddx_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddx_num_expr, 'numpy')
    ddphi_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddphi_num_expr, 'numpy')
    ddtheta_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddtheta_num_expr, 'numpy')
    print ("ddx: ", ddx_num, "\nddphi: ", ddphi_num, "\nddtheta: ", ddtheta_num)


    return ddx_num, ddphi_num, ddtheta_num


def lagrangian_fast(m_1_data, m_2_data, l_data, R_data):
    # Defining paramters and symbols
    t = symbols('t', real=True, positive=True)
    k_1 = symbols('k_1', real=True, positive=True)
    k_2 = symbols('k_2', real=True, positive=True)
    kt_1 = symbols('kt_1', real=True, positive=True)
    kt_2 = symbols('kt_2', real=True, positive=True)
    # General coordinates
    x = Function('x')(t)
    phi = Function('phi')(t)
    theta = Function('theta')(t)
    r = Function('r')(t)    # gain
    # Constans 
    m_1, m_2, l, R = symbols("m_1, m_2, l, R", real=True, positive=True)
    data = [(m_1, m_1_data), (m_2, m_2_data), (l, l_data), (R, R_data)]

    # Moment of inertia
    theta_1 = (1/12)*m_1*l**2
    theta_2 = (1/2)*m_2*R**2

    # Angular velocity
    omega_1 = phi.diff(t)
    omega_2 = phi.diff(t) + theta.diff(t)
    omega_21 = theta.diff(t)

    # Velocity
    r_S01 = Matrix([(l/2)*sin(phi), (l/2)*cos(phi), 0])
    v_S0 = Matrix([x.diff(t), 0, 0])
    omega_v1 = Matrix([0, 0, omega_1])
    v_v1 = v_S0 + omega_v1.cross(r_S01)
    # print ("Velocity of S1:", v_v1)

    r_S0C = Matrix([l*sin(phi), l*cos(phi), 0])
    omega_v2 = Matrix([0, 0, omega_2])  # not needed
    omega_v21 = Matrix([0,0,omega_21])
    v_C = v_S0 + omega_v1.cross(r_S0C)
    r_CS2 = Matrix([R*sin(theta), R*cos(theta), 0])
    v_v2 = v_C + omega_v21.cross(r_CS2)
    # print ("Velocity of S2: ", v_v2)
    # print("Magnitude of v_1 and v_2: ", v_1, "and ", v_2)

    # Displacements for k
    x_1 = x
    x_2 = x + (3/4)*l*sin(phi)

    # Angles for k_t
    phi_1 = phi
    phi_2 = theta

    # Kinetic energy
    T = (1/2)*(m_1*v_v1.dot(v_v1) + m_2*v_v2.dot(v_v2) + theta_1 * omega_1**2 + theta_2*omega_2**2)

    # Potential energy 
    U = (1/2)*(k_1*(r-x_1)**2 + k_2*(r-x_2)**2 + kt_1*phi_1**2 + kt_2*phi_2**2)

    # Derivatives
    dx = x.diff(t)
    ddx = dx.diff(t)
    dphi=phi.diff(t)
    ddphi=dphi.diff(t)
    dtheta = theta.diff(t)
    ddtheta =dtheta.diff(t)
    dT_dx = T.diff(x)
    dT_dxd = T.diff(dx)
    dT_dxdt = dT_dxd.diff(t)
    dU_dx = U.diff(x)
    dT_dphi = T.diff(phi)
    dT_dphid = T.diff(dphi)
    dT_dphidt = dT_dphid.diff(t)
    dU_dphi = U.diff(phi)
    dT_dtheta = T.diff(theta)
    dT_dthetad = T.diff(dtheta)
    dT_dthetadt = dT_dthetad.diff(t)
    dU_dtheta = U.diff(theta)
    ###
    eq1 = dT_dxdt - dT_dx + dU_dx
    eq2 = dT_dphidt - dT_dphi + dU_dphi
    eq3 = dT_dthetadt - dT_dtheta + dU_dtheta
    # Simplifying equations
    eq1_simplified = sp.simplify(eq1)
    eq2_simplified = sp.simplify(eq2)
    eq3_simplified = sp.simplify(eq3)
    eqs = sp.Matrix([eq1_simplified, eq2_simplified, eq3_simplified])
    variables = sp.Matrix([ddx, ddphi, ddtheta, r])
    substitutions = {m_1: m_1_data, m_2: m_2_data, l: l_data, R: R_data}
    eq1_subs = eq1_simplified.subs(substitutions)
    eq2_subs = eq2_simplified.subs(substitutions)
    eq3_subs = eq3_simplified.subs(substitutions)
    eqs_subs = sp.Matrix([eq1_subs, eq2_subs, eq3_subs])
    # Solving the system
    # solution = sp.solve(eqs, variables)
    solution = sp.solve(eqs_subs, variables)
    print("Solution of differential equation:", solution)
    # Substituting numerical values for constants
    # solution_with_values = {key: value.subs(data).evalf() for key, 
    #                         value in solution.items()}
    
    # # Conversion to numerical functions
    # ddx_num_expr = solution_with_values[ddx]
    # ddphi_num_expr = solution_with_values[ddphi]
    # ddtheta_num_expr = solution_with_values[ddtheta]
    # Conversion to numerical functions
    ddx_num_expr = solution[ddx]
    ddphi_num_expr = solution[ddphi]
    ddtheta_num_expr = solution[ddtheta]
    ddx_num_expr_simplified = sp.simplify(ddx_num_expr)
    ddphi_num_expr_simplified = sp.simplify(ddphi_num_expr)
    ddtheta_num_expr_simplified = sp.simplify(ddtheta_num_expr)
    # lambdify még nem biztos, hogy jó
    # ddx_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddx_num_expr, 'numpy')
    # ddphi_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddphi_num_expr, 'numpy')
    # ddtheta_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddtheta_num_expr, 'numpy')
    # print ("ddx: ", ddx_num, "\nddphi: ", ddphi_num, "\nddtheta: ", ddtheta_num)

    ddx_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddx_num_expr_simplified, 'numpy')
    ddphi_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddphi_num_expr_simplified, 'numpy')
    ddtheta_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddtheta_num_expr_simplified, 'numpy')
    print ("ddx: ", ddx_num, "\nddphi: ", ddphi_num, "\nddtheta: ", ddtheta_num)

    return ddx_num, ddphi_num, ddtheta_num

def lagrangian_fast2(m_1_data, m_2_data, l_data, R_data):
    # Defining paramters and symbols
    t = symbols('t', real=True, positive=True)
    k_1 = symbols('k_1', real=True, positive=True)
    k_2 = symbols('k_2', real=True, positive=True)
    kt_1 = symbols('kt_1', real=True, positive=True)
    kt_2 = symbols('kt_2', real=True, positive=True)
    # General coordinates
    x = Function('x')(t)
    phi = Function('phi')(t)
    theta = Function('theta')(t)
    r = Function('r')(t)    # gain
    # Constans 
    m_1, m_2, l, R = symbols("m_1, m_2, l, R", real=True, positive=True)
    data = [(m_1, m_1_data), (m_2, m_2_data), (l, l_data), (R, R_data)]

    # Moment of inertia
    theta_1 = (1/12)*m_1*l**2
    theta_2 = (1/2)*m_2*R**2

    # Angular velocity
    omega_1 = phi.diff(t)
    omega_2 = phi.diff(t) + theta.diff(t)
    omega_21 = theta.diff(t)

    # Velocity
    r_S01 = Matrix([(l/2)*sin(phi), (l/2)*cos(phi), 0])
    v_S0 = Matrix([x.diff(t), 0, 0])
    omega_v1 = Matrix([0, 0, omega_1])
    v_v1 = v_S0 + omega_v1.cross(r_S01)
    # print ("Velocity of S1:", v_v1)

    r_S0C = Matrix([l*sin(phi), l*cos(phi), 0])
    omega_v2 = Matrix([0, 0, omega_2])  # not needed
    omega_v21 = Matrix([0,0,omega_21])
    v_C = v_S0 + omega_v1.cross(r_S0C)
    r_CS2 = Matrix([R*sin(theta), R*cos(theta), 0])
    v_v2 = v_C + omega_v21.cross(r_CS2)
    # print ("Velocity of S2: ", v_v2)
    # print("Magnitude of v_1 and v_2: ", v_1, "and ", v_2)

    # Displacements for k
    x_1 = x
    x_2 = x + (3/4)*l*sin(phi)

    # Angles for k_t
    phi_1 = phi
    phi_2 = theta

    # Kinetic energy
    T = (1/2)*(m_1*v_v1.dot(v_v1) + m_2*v_v2.dot(v_v2) + theta_1 * omega_1**2 + theta_2*omega_2**2)

    # Potential energy 
    U = (1/2)*(k_1*(r-x_1)**2 + k_2*(r-x_2)**2 + kt_1*phi_1**2 + kt_2*phi_2**2)

    # Derivatives
    dx = x.diff(t)
    ddx = dx.diff(t)
    dphi=phi.diff(t)
    ddphi=dphi.diff(t)
    dtheta = theta.diff(t)
    ddtheta =dtheta.diff(t)
    dT_dx = T.diff(x)
    dT_dxd = T.diff(dx)
    dT_dxdt = dT_dxd.diff(t)
    dU_dx = U.diff(x)
    dT_dphi = T.diff(phi)
    dT_dphid = T.diff(dphi)
    dT_dphidt = dT_dphid.diff(t)
    dU_dphi = U.diff(phi)
    dT_dtheta = T.diff(theta)
    dT_dthetad = T.diff(dtheta)
    dT_dthetadt = dT_dthetad.diff(t)
    dU_dtheta = U.diff(theta)
    ###
    eq1 = dT_dxdt - dT_dx + dU_dx
    eq2 = dT_dphidt - dT_dphi + dU_dphi
    eq3 = dT_dthetadt - dT_dtheta + dU_dtheta
    # Simplifying equations
    eq1_simplified = sp.simplify(eq1)
    eq2_simplified = sp.simplify(eq2)
    eq3_simplified = sp.simplify(eq3)
    eqs = sp.Matrix([eq1_simplified, eq2_simplified, eq3_simplified])
    variables = sp.Matrix([ddx, ddphi, ddtheta, r])
    substitutions = {m_1: m_1_data, m_2: m_2_data, l: l_data, R: R_data}
    eq1_subs = eq1_simplified.subs(substitutions)
    eq2_subs = eq2_simplified.subs(substitutions)
    eq3_subs = eq3_simplified.subs(substitutions)
    eqs_subs = sp.Matrix([eq1_subs, eq2_subs, eq3_subs])
    # Solving the system
    # solution = sp.solve(eqs, variables)
    solution = sp.solve(eqs_subs, variables)
    print("Solution of differential equation calculateed")
    # print("Solution of differential equation:", solution)
    # Substituting numerical values for constants
    # solution_with_values = {key: value.subs(data).evalf() for key, 
    #                         value in solution.items()}
    
    # # Conversion to numerical functions
    # ddx_num_expr = solution_with_values[ddx]
    # ddphi_num_expr = solution_with_values[ddphi]
    # ddtheta_num_expr = solution_with_values[ddtheta]
    # Conversion to numerical functions
    ddx_num_expr = solution[ddx]
    ddphi_num_expr = solution[ddphi]
    ddtheta_num_expr = solution[ddtheta]
    print("expr solution complete")
    print("ddx_num_expr")
    print(ddx_num_expr)
    print("ddphi_num_expr")
    print(ddphi_num_expr)
    print("ddtheta_num_expr")
    print(ddtheta_num_expr)
    # lambdify még nem biztos, hogy jó
    ddx_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddx_num_expr, 'numpy')
    ddphi_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddphi_num_expr, 'numpy')
    ddtheta_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), ddtheta_num_expr, 'numpy')
    print ("ddx: ", ddx_num, "\nddphi: ", ddphi_num, "\nddtheta: ", ddtheta_num)

    
    return ddx_num, ddphi_num, ddtheta_num


# gives back displacement and velocity values that can be substituted back to the equation 
def num_solver_for_disp(ddx_num, gain_input, time_input, y0, max_step, k):

    gain_interpolated = interp1d(time_input, gain_input)
    k = [float(k) for t in time_input]
    k = np.array(k)
    k_interpolated = interp1d(time_input, k)

    # Defineing the system of ODEs
    def system(t, y):
        x_val, dx_val = y
        r = gain_interpolated(t)
        k = k_interpolated(t)
        ddx_val = float(ddx_num(t, x_val, dx_val, r, k))
        # ddphi_val = float(ddphi_num(t, x_val, dx_val, phi_val, dphi_val, r))
        return [dx_val, ddx_val]

    solver = RK45(system, time_input[0], y0, time_input[-1], max_step=max_step)
    t_val = []
    y_val = []

    for i in range(1400):
        solver.step()
        t_val.append(solver.t)
        y_val.append(solver.y)

        if solver.status == 'finished':
            break

    t_val = np.array(t_val)
    y_val = np.array(y_val)

    x_val = y_val[:, 0]
    dx_val = y_val[:, 1]
    
    return x_val, dx_val

def num_solver(ddx_num, ddphi_num, ddtheta_num, gain_input, time_input, y0, k_1, kt_1, k_2, kt_2):

    gain_interpolated = interp1d(time_input, gain_input)
    k_1 = [float(k_1) for t in time_input]
    k_1 = np.array(k_1)
    k_1_interpolated = interp1d(time_input, k_1)
    k_2 = [float(k_2) for t in time_input]
    k_2 = np.array(k_2)
    k_2_interpolated = interp1d(time_input, k_2)
    kt_1 = [float(kt_1) for t in time_input]
    kt_1 = np.array(kt_1)
    kt_1_interpolated = interp1d(time_input, kt_1)
    kt_2 = [float(kt_2) for t in time_input]
    kt_2 = np.array(kt_2)
    kt_2_interpolated = interp1d(time_input, kt_2)

    # Defineing the system of ODEs
    def system(t, y):
        x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val = y
      
    # Continue with the rest of the function
        r = gain_interpolated(t)
        k_1 = k_1_interpolated(t)
        k_2 = k_2_interpolated(t)
        kt_1 = kt_1_interpolated(t)
        kt_2 = kt_2_interpolated(t)
        # print(f"t: {t}, x_val: {x_val}, dx_val: {dx_val}, phi_val: {phi_val}, dphi_val: {dphi_val}, theta_val: {theta_val}, dtheta_val: {dtheta_val}")
        try:
            ddphi_val = float(ddphi_num(t, x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val, r, k_1, k_2, kt_1, kt_2))
        except Exception as e:
            print(f"Error in ddphi_num: {e}")
            raise
        ddx_val = float(ddx_num(t, x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val, r, k_1, k_2, kt_1, kt_2))
        ddphi_val = float(ddphi_num(t, x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val, r, k_1, k_2, kt_1, kt_2))
        ddtheta_val = float(ddtheta_num(t, x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val, r, k_1, k_2, kt_1, kt_2))
        return [dx_val, ddx_val, dphi_val, ddphi_val, dtheta_val, ddtheta_val]

    solution = solve_ivp( system, ( time_input[0], time_input[-1] ), y0, 'RK45', time_input )
    t_val = solution.t
    x_val = solution.y[0]
    dx_val = solution.y[1]
    phi_val = solution.y[2]
    dphi_val = solution.y[3]
    theta_val = solution.y[4]
    dtheta_val = solution.y[5]

    return x_val, dx_val, phi_val, dphi_val, theta_val, dtheta_val, t_val


def acceleration_substituted(ddx_num, ddphi_num, ddtheta_num, gain_input, time_input, y0,  k_1, k_2, kt_1, kt_2):
    x_values, dx_values, phi_values, dphi_values, theta_values, dtheta_values, _ = num_solver(ddx_num, ddphi_num, ddtheta_num, gain_input, time_input, y0, k_1, k_2, kt_1, kt_2)

    # (t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2)
    ddx_subst = ddx_num(time_input, x_values, dx_values, phi_values, dphi_values, theta_values, dtheta_values, gain_input,k_1, k_2, kt_1, kt_2)
    ddphi_subst = ddphi_num(time_input, x_values, dx_values, phi_values, dphi_values, theta_values, dtheta_values, gain_input,k_1, k_2, kt_1, kt_2)
    ddtheta_subst = ddtheta_num(time_input, x_values, dx_values, phi_values, dphi_values, theta_values, dtheta_values, gain_input,k_1, k_2, kt_1, kt_2)

    return ddx_subst, ddphi_subst, ddtheta_subst

def relative_acceleration_formula(a_Ax, a_Ay, omega, r_ABx, r_ABy, domega):

    a_A = np.array([a_Ax, a_Ay, np.zeros_like(a_Ax)])  
    eps = np.array([np.zeros_like(domega), np.zeros_like(domega), domega])  
    omega_v = np.array([np.zeros_like(omega), np.zeros_like(omega), omega]) 
    r_AB = np.array([r_ABx, r_ABy, np.zeros_like(r_ABx)]) 
    

    a_B = (
        a_A +
        np.cross(eps.T, r_AB.T).T +
        np.cross(omega_v.T, np.cross(omega_v.T, r_AB.T)).T
    )

    return a_B  

def calculate_acceleration_components_for_all_body_parts( ddx_num, ddphi_num, ddtheta_num, sled_disp, sim_time, y0, k_1, k_2, kt_1, kt_2, R, l ):

    x_computed, dx_computed, phi_computed, dphi_computed, theta_computed, dtheta_computed, t_computed = \
        num_solver(ddx_num, ddphi_num, ddtheta_num, sled_disp, sim_time, y0, k_1, kt_1, k_2, kt_2) # displacements in mm, velocities in mm/s
    
    a_pelvis_x, ddphi_computed, ddtheta_computed = \
        acceleration_substituted(ddx_num, ddphi_num, ddtheta_num, sled_disp, sim_time, y0,  k_1, k_2, kt_1, kt_2)

    a_pelvis_y = np.zeros_like( a_pelvis_x )

    # CHEST acceleration
    r_pch_x = (3/4)*l*np.sin( phi_computed ) # mm
    r_pch_y = (3/4)*l*np.cos( phi_computed ) # mm

    a_chest = relative_acceleration_formula( a_pelvis_x, a_pelvis_y, dphi_computed, r_pch_x, r_pch_y, ddphi_computed )
    a_chest_x = a_chest[0, :]  
    a_chest_y = a_chest[1, :]

    # NECK acceleration
    r_pn_x = l*np.sin( phi_computed )  # mm
    r_pn_y = l*np.cos( phi_computed )  # mm

    a_neck = relative_acceleration_formula( a_pelvis_x, a_pelvis_y, dphi_computed, r_pn_x, r_pn_y, ddphi_computed )
    a_neck_x = a_neck[0, :]  
    a_neck_y = a_neck[1, :]

    # HEAD acceleration
    r_nh_x = R*np.sin( theta_computed ) # mm
    r_nh_y = R*np.cos( theta_computed ) # mm

    a_head = relative_acceleration_formula( a_neck_x, a_neck_y, dtheta_computed, r_nh_x, r_nh_y, ddtheta_computed )
    a_head_x = a_head[0, :]  
    a_head_y = a_head[1, :]

    return a_pelvis_x, a_pelvis_y, a_chest_x, a_chest_y, a_head_x, a_head_y



def difference_calc(data_measured, data_computed):
    diff = data_measured - data_computed
    return diff