from seaice import *

timescale = 10
timestep = 1
dumpfreq = 10
dirname = "./output/test/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = StrainRateTensor(timestepping=timestepping, number_of_triangles=35, output=output, params=params,
                       solver_params=solver)

evp = Evp(timestepping=timestepping, number_of_triangles=35, output=output, params=params, solver_params=solver)

bt = BoxTest(timestepping=timestepping, number_of_triangles=35, output=output, params=params, solver_params=solver)


def test_srt():
    assert srt


def test_evp():
    assert evp


def test_bt():
    assert bt
