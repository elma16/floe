from seaice import *
from firedrake import *
import numpy as np

timestep = 1
dumpfreq = 10 ** 6
timescale = 10

dirname = "test.pvd"
number_of_triangles = [5, 10, 20, 40, 100]

length = 5 * 10 ** 5
pi_x = pi / length
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()
error_values = []

for values in number_of_triangles:
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
    conditions = {'bc': [0], 'ic': [v_exp]}
    srt = ViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                         params=params, solver_params=solver, stabilised=False, simple=True, family='CR')

    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep

    error_values.append(Error.compute(srt.u1, v_exp))

error_slope = float(format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f'))


def test_srt_initial_value():
    assert error_slope + 2 < 0.01
