import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from seaice.models import *



dirname = "./output/EVP/u.pvd"

timestepping = TimesteppingParameters(timescale=100, timestep=10)
output = OutputParameters(dirname=dirname, dumpfreq=10)
params = SeaIceParameters()

evp = Evp(number_of_triangles=30, output=output, params=params, timescale=100, timestep=10)

t = 0
evp.solve(t)
evp.update(t)
evp.dump(t)
