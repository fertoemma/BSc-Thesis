import lmfit
import numpy as np

def f_squared( x, a, b ):
    return a * x**2 + b

def f_qubed( x, a, b ):
    return a * x**3 + b

eval_points = np.linspace( 0, 4, 200 )

a1_true = 3.2
b1_true = 2.3
a2_true = 0.5
b2_true = 1.5

noise_squared = f_squared( eval_points, a1_true, b1_true ) + np.random.random( len(eval_points) )
noise_qubed = f_qubed( eval_points, a2_true, b2_true ) + np.random.random( len(eval_points) )

def obj_func( params, eval_points, data1, data2 ):

    a1 = params["a1"]
    b1 = params["b1"]
    a2 = params["a2"]
    b2 = params["b2"]

    return ( f_squared( eval_points, a1, b1 ) - data1 ) ** 2 + ( f_qubed( eval_points, a2, b2 ) - data2 ) ** 2

params_to_fit = lmfit.Parameters()
params_to_fit.add( "a1", value=1.0 )
params_to_fit.add( "b1", value=1.0 )
params_to_fit.add( "a2", value=1.0 )
params_to_fit.add( "b2", value=1.0 )



results = lmfit.minimize( obj_func, params_to_fit, "least_squares", args=( eval_points, noise_squared, noise_qubed ) )
print( results.params )
print( results.params["a1"] )
print( results.params["b1"] )
print( results.params["a2"] )
print( results.params["b2"] )
print( results.params["a1"].value )
print( results.params["b1"].value )
print( results.params["a2"].value )
print( results.params["b2"].value )