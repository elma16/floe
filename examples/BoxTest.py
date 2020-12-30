from seaice2 import *

dirname = "./output/box_test/u.pvd"

timestepping = TimesteppingParameters(timescale=10, timestep=1)
output = OutputParameters(dirname=dirname, dumpfreq=20)
params = SeaIceParameters()

bt = BoxTest(number_of_triangles=30, params=params, stabilised=False, timescale=10, timestep=1)

t = 0
bt.solve(t)
bt.update(t)
bt.dump(t)
