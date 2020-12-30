import sys
from seaice import *

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
else:
    timestep = 1

if '--long' in sys.argv:
    timescale = 100
else:
    timescale = 10

dirname = "./output/EVP/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=10)
params = SeaIceParameters()

evp = Evp(number_of_triangles=35,params=params,timestepping=timestepping)

t = 0
evp.solve(t)
evp.update(t)
evp.dump(t)
