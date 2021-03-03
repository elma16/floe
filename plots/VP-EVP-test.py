from seaice import *
from firedrake import *
from pathlib import Path

Path("./output/evp").mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP

Test 1 : Compare VP and EVP velocity convergence against a stationary solution.
v = [-sin(pi_x x)sin(pi_x y),-sin(pi_x x)sin(pi_x y)] 
'''
timestep = 10
dumpfreq = 10 ** 3
timescale = timestep * dumpfreq

stabilise = False

dirname = "./output/vp_evp/u_timescale={}_timestep={}_stabilised={}_real.pvd".format(timescale, timestep, stabilise)
title = "VP EVP Plot"
diagnostic_dirname = "./output/vp_evp/evp.nc"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x), -sin(pi_x * x)])

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': [0, 1, 1],
              'ic': [v_exp, x / length, as_matrix([[0, 0], [0, 0]])],
              'ocean_curr': ocean_curr,
              'geo_wind': Constant(as_vector([0, 0]))}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                            params=params, solver_params=solver, stabilised=stabilise)

vp = ViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                    params=params, solver_params=solver, stabilised=stabilise, simple=False)

t = 0

while t < timescale - 0.5 * timestep:
    evp.solve(evp.usolver)
    evp.update(evp.w0, evp.w1)
    evp.dump(evp.u1, evp.s1, t=t)
    t += timestep
    evp.progress(t)
