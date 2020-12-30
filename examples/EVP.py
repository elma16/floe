import sys
from seaice import *

# TEST 2 : EVP

timestep = 10**(-1)

timescale = 10

dirname = "./output/EVP/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=10)
solver = SolverParameters()
params = SeaIceParameters()

evp = Evp(number_of_triangles=35, params=params, timestepping=timestepping, output=output, solver_params=solver)

t = 0
evp.solve(t)
evp.update(t)
evp.dump(t)
