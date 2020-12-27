import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config.config import *
from models.strain_rate_tensor import *

dirname = "./test.pvd"

timestepping = TimesteppingParameters(timescale=10, timestep=1)
output = OutputParameters(dirname=dirname, dumpfreq=20)
params = SeaIceParameters()

StrainRateTensor(timestepping=timestepping, number_of_triangles=30, output=output, params=params, stabilised=False,
                 shape=None, transform_mesh=False)

# StrainRateTensor.solve()
# StrainRateTensor.update()
# StrainRateTensor.dump()
