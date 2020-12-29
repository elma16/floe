import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from seaice.models import *

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
else:
    timestep = 1

if '--long' in sys.argv:
    timescale = 100
else:
    timescale = 10

dirname = "./output/strain_rate_tensor/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=10)
params = SeaIceParameters()

srt = StrainRateTensor(timestepping=timestepping, number_of_triangles=35, output=output, params=params,
                       stabilised=0,
                       shape=None, transform_mesh=False)

t = 0
srt.solve(t)
srt.update(t)
srt.dump(t)
