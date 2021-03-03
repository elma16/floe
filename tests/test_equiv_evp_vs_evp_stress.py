from seaice import *
from firedrake import *
import numpy as np

timestep = 1
dumpfreq = 10
timescale = 10

dirname = "test.pvd"

number_of_triangles = [5, 10, 20, 40, 100]
length = 5 * 10 ** 5
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

error_values = []

for values in number_of_triangles:
    print(values)
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
    forcing = [ocean_curr]
    conditions = {'bc': [0, 1, 1], 'ic': [0, x / length, as_matrix([[0, 0], [0, 0]])], 'ocean_curr': ocean_curr}
    evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions,
                                timestepping=timestepping, output=output, params=params, solver_params=solver,
                                stabilised=False)
    evps = ElasticViscousPlasticStress(mesh=mesh, length=length, conditions=conditions,
                                       timestepping=timestepping, output=output, params=params, solver_params=solver,
                                       stabilised=False)

    t = 0

    while t < timescale - 0.5 * timestep:
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        evps.solve(evps.usolver)
        evps.update(evps.u0, evps.u1)
        t += timestep
    u1, s1 = evp.w1.split()
    error_values.append(Error.compute(evps.u1, u1))

error_slope = format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f')

def test_srt_initial_value():
    assert error_slope + 2 < 0.01