import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import equation_solver_3DoF as solv

sys.path.append(r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling\function_files')
import krc_reader
import filterSAEJ211
import data_saver as data

sp.init_printing()
from sympy import symbols

import numpy as np
import sympy as sp
sp.init_printing()
from sympy import symbols, Function, diff, Matrix, sin, cos, lambdify
import scipy.integrate as integrate
from scipy.integrate import RK45
from scipy.interpolate import interp1d
import sympy as sp
from sympy import Function, Matrix, symbols, sin, cos
from sympy.utilities.lambdify import lambdify



# path_EU = os.path.join( os.path.dirname( os.path.abspath('') ), 'Dataset', 'EU2' )
# path_US = os.path.join( os.path.dirname( os.path.abspath('') ), 'Dataset', 'US' )
base_path = r'C:\Users\emmaf\Documents\7. félév\SZAKDOLGOZAT\osd_dummy_modeling\Dataset'
# path_EU = os.path.join( os.path.abspath(''), 'Dataset', 'EU2' )
path_EU = os.path.join(base_path, 'EU2')
path_US = os.path.join(base_path, 'US')
# path_US = os.path.join( os.path.abspath(''), 'Dataset', 'US' )

# sim_repl_lib_paths = [ path_US  ]
sim_repl_lib_paths = [ path_EU, path_US  ]

krc_data_reader = krc_reader.KeyResultCurveDataReader( sim_repl_lib_paths )
krc_data_reader.load_replications_results( moving_average_win_len=10 )
krc_data_reader.get_channels_with_missing_data()

krc_data_reader.remove_channels_from_all_replications( [ '11CHST0000H3DSX0' ] )
krc_data_reader.get_channels_with_missing_data()

krc_data_reader.remove_replications( [ ('US', 'SO000'), ('US', 'SO019'), ('US', 'SO095'), ('EU2', 'SO095') ] ) #('US', 'SO024'), ('US', 'SO056'),
krc_data_reader.get_channels_with_missing_data()

channels_in_data = krc_data_reader.get_available_channels()
print( channels_in_data )


krc_data_reader.get_available_replications()

sing_rep_res = krc_data_reader.get_single_replication_results( ('EU2', 'SO066') )
plt.plot( sing_rep_res['10SLEDFRMI00ACX0'] )


# Filter
data = sing_rep_res['10SLEDFRMI00ACX0']
# data = data
filtered_data = filterSAEJ211.filterSAEJ211(data, 6, 0.001)
# time = np.linspace(0, 0.00014, 1400)
plt.plot(filtered_data)

# Given acceleration
acceleration_data = np.array(filtered_data) # mm/s^2
sled_acceleration_data = acceleration_data/10**12 # mm/us^2
time = np.linspace(0, 0.00014*10**6, 1400) # microsec
# Initial conditions
x_0 = 0
v_0 =0
y0 = [x_0, v_0]
max_step = 0.0000001*10**6
# Sled displacement
gain, t_values = solv.sled_acceleration_to_displacement(sled_acceleration_data, time, y0, max_step)
    
# Plotting the results
plt.figure()
plt.plot(t_values, gain)
plt.xlabel('Time [us]')
plt.ylabel('Gain [mm]')
plt.title("Gain from sled acceleration data")
plt.grid() # kikapcsolható
plt.show()

import sympy as sp
from sympy import Function, Matrix, symbols, sin, cos
from sympy.utilities.lambdify import lambdify

# Paraméterek és szimbólumok
m_1, m_2, l, R = symbols("m_1 m_2 l R")
k_1, k_2, kt_1, kt_2 = symbols("k_1 k_2 kt_1 kt_2")
t = symbols("t")
x = Function("x")(t)
phi = Function("phi")(t)
theta = Function("theta")(t)
r = Function("r")(t)

# Tehetetlenségi nyomaték
theta_1 = (1 / 12) * m_1 * l**2
theta_2 = (1 / 2) * m_1 * R**2

# Sebességek és szögsebességek
dx = x.diff(t)
ddx = dx.diff(t)
dphi = phi.diff(t)
ddphi = dphi.diff(t)
dtheta = theta.diff(t)
ddtheta = dtheta.diff(t)

r_S01 = Matrix([(l / 2) * sin(phi), (l / 2) * cos(phi), 0])
v_S0 = Matrix([dx, 0, 0])
omega_v1 = Matrix([0, 0, dphi])
v_v1 = v_S0 + omega_v1.cross(r_S01)

r_S0C = Matrix([l * sin(phi), l * cos(phi), 0])
omega_v2 = Matrix([0, 0, dphi + dtheta])
v_C = v_S0 + omega_v1.cross(r_S0C)
r_CS2 = Matrix([R * sin(phi + theta), R * cos(phi + theta), 0])
v_v2 = v_C + omega_v2.cross(r_CS2)

# Kinetikus energia (T)
T = (1 / 2) * (m_1 * v_v1.dot(v_v1) + m_2 * v_v2.dot(v_v2) + theta_1 * dphi**2 + theta_2 * (dphi + dtheta)**2)

# Potenciális energia (U)
x_1 = x
x_2 = x + (3 / 4) * l * sin(phi)
phi_1 = phi
phi_2 = theta
U = (1 / 2) * (k_1 * (r - x_1)**2 + k_2 * (r - x_2)**2 + kt_1 * phi_1**2 + kt_2 * phi_2**2)

# Lagrange-egyenletek
L = T - U
eq1 = sp.Eq(L.diff(dx).diff(t) - L.diff(x), 0)
eq2 = sp.Eq(L.diff(dphi).diff(t) - L.diff(phi), 0)
eq3 = sp.Eq(L.diff(dtheta).diff(t) - L.diff(theta), 0)

# Egyenletek megoldása
variables = [ddx, ddphi, ddtheta]
eqs = [eq1, eq2, eq3]
solutions = sp.solve(eqs, variables)

# Numerikus helyettesítés
data = [(m_1, 34), (m_2, 4), (l, 9.96), (R, 9 / 100), (k_1, 0.9), (k_2, 1.1), (kt_1, 20), (kt_2, 30)]
subs_data = {k: v for k, v in data}
solutions_with_values = {k: v.subs(subs_data).evalf() for k, v in solutions.items()}

# Gyorsított lambdify függvények
ddx_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r), solutions_with_values[ddx], "numpy")
ddphi_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r), solutions_with_values[ddphi], "numpy")
ddtheta_num = lambdify((t, x, dx, phi, dphi, theta, dtheta, r), solutions_with_values[ddtheta], "numpy")

print("ddx:", ddx_num)
print("ddphi:", ddphi_num)
print("ddtheta:", ddtheta_num)




# # Paraméterek definiálása
# m_1, m_2, l, R = symbols("m_1 m_2 l R")
# data = [(m_1, 34), (m_2, 4), (l, 9.96), (R, 9 / 100)]
# t = symbols("t")
# k_1, k_2, kt_1, kt_2 = symbols("k_1 k_2 kt_1 kt_2")

# # Általános koordináták
# x = Function("x")(t)
# phi = Function("phi")(t)
# theta = Function("theta")(t)
# r = Function("r")(t)

# # Tehetetlenségi nyomaték
# theta_1 = (1 / 12) * m_1 * l**2
# theta_2 = (1 / 2) * m_1 * R**2

# # Sebességek és szöggyorsulások
# omega_1 = phi.diff(t)
# omega_2 = phi.diff(t) + theta.diff(t)

# # Potenciális és kinetikus energia
# T = (
#     (1 / 2) * m_1 * ((l / 2) * phi.diff(t))**2
#     + (1 / 2) * m_2 * ((l) * phi.diff(t))**2
#     + (1 / 2) * theta_1 * omega_1**2
#     + (1 / 2) * theta_2 * omega_2**2
# )

# U = (
#     (1 / 2) * k_1 * (r - x)**2
#     + (1 / 2) * k_2 * (r - (x + (3 / 4) * l * sin(phi)))**2
#     + (1 / 2) * kt_1 * phi**2
#     + (1 / 2) * kt_2 * theta**2
# )

# # Lagrange-egyenletek
# dx = x.diff(t)
# ddx = dx.diff(t)
# dphi = phi.diff(t)
# ddphi = dphi.diff(t)
# dtheta = theta.diff(t)
# ddtheta = dtheta.diff(t)
# ddx = x.diff(t, 2)
# ddphi = phi.diff(t, 2)
# ddtheta = theta.diff(t, 2)
# L = T - U
# eq1 = sp.Eq(L.diff(dx).diff(t) - L.diff(x), 0)
# eq2 = sp.Eq(L.diff(dphi).diff(t) - L.diff(phi), 0)
# eq3 = sp.Eq(L.diff(dtheta).diff(t) - L.diff(theta), 0)
# print("Equation 1:", eq1)
# print("Equation 2:", eq2)
# print("Equation 3:", eq3)
# # Numerikus helyettesítés
# eqs = sp.Matrix([eq1, eq2, eq3])
# variables = sp.Matrix([ddx, ddphi, ddtheta])
# subs_data = {k: v for k, v in data}
# solutions = sp.solve(eqs, variables)
# print("SOlutions:", solutions)
# # Kiértékelés numerikus formában
# solutions_with_values = {k: v.subs(subs_data).evalf() for k, v in solutions.items()}
# ddx_num_expr = solutions_with_values[ddx]
# ddphi_num_expr = solutions_with_values[ddphi]
# ddtheta_num_expr = solutions_with_values[ddtheta]

# ddx_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddx], "numpy")
# ddphi_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddphi], "numpy")
# ddtheta_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddtheta], "numpy")


# # # Gyorsított lambdify
# # ddx_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddx], "numpy")
# # ddphi_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddphi], "numpy")
# # ddtheta_func = lambdify((t, x, dx, phi, dphi, theta, dtheta, r, k_1, k_2, kt_1, kt_2), solutions_with_values[ddtheta], "numpy")
