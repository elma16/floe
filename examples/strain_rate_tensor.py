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

dirname = "./output/strain_rate_tensor/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

plot_dirname = "./plots/strain_rate_energy.png"

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=10)
solver = SolverParameters()
params = SeaIceParameters()

srt = StrainRateTensor(timestepping=timestepping, number_of_triangles=35, output=output, params=params,
                       solver_params=solver)

t = 0
srt.solve(t)
srt.update(t)
srt.dump(t)

err = Error(model=srt, dirname=plot_dirname)


