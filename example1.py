import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models import *

dirname = "./output/strain_rate_tensor/u.pvd"

timestepping = TimesteppingParameters(timescale=10, timestep=1)
output = OutputParameters(dirname=dirname, dumpfreq=20)
params = SeaIceParameters()

srt = StrainRateTensor(timescale=10, timestep=1, number_of_triangles=30, output=output, params=params, stabilised=False,
                       shape=None, transform_mesh=False)

t = 0
for i in range(10):
    srt.solve(t)
    srt.update(t)
    srt.dump(t)
