import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from seaice.models import *

dirname = "./output/box_test/u.pvd"

timestepping = TimesteppingParameters(timescale=10, timestep=1)
output = OutputParameters(dirname=dirname, dumpfreq=20)
params = SeaIceParameters()

bt = BoxTest(number_of_triangles=30, output=output, params=params, stabilised=False, timescale=10, timestep=1)

t = 0
bt.solve(t)
bt.update(t)
bt.dump(t)
