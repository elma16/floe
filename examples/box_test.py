import sys
from seaice import *

# TEST 3 : BOX TEST

if '--test' in sys.argv:
    timestep = 600
    number_of_triangles = 71
    day = 60 * 60 * 24
    month = 31 * day
    timescale = month
else:
    timescale = 100
    number_of_triangles = 30
    timestep = 1

dirname = "./output/box_test/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

plot_dirname = "./plots/box_test_energy.png"

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=10)
solver = SolverParameters()
params = SeaIceParameters()

bt = BoxTest(number_of_triangles=number_of_triangles, params=params, solver_params=solver, output=output,
             timestepping=timestepping)

t = 0
bt.solve(t)
bt.update(t)
bt.dump(t)
