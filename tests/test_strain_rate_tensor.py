import pytest
from seaice import *
from firedrake import *

timestep = 1
dumpfreq = 10
timescale = 10
number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
bcs_values = [0]
ics_values = [0]

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, ics_values=ics_values, timestepping=timestepping,
                     output=output, params=params, solver_params=solver)


def test_srt_initial_value():
    assert srt == 0
