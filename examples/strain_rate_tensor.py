import sys
from seaice import *
from time import time

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10**5
else:
    timestep = 1
    dumpfreq = 10

if '--long' in sys.argv:
    timescale = 100
else:
    timescale = 10

dirname = "./output/strain_rate_tensor/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

plot_dirname = "./plots/strain_rate_energy.png"

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = StrainRateTensor(timestepping=timestepping, number_of_triangles=35, output=output, params=params,
                       solver_params=solver)

t = 0
start = time()
while t < timescale - 0.5 * timestep:
    srt.solve()
    srt.update()
    srt.dump(t)
    t += timestep
    srt.progress(t)
end = time()
print(end - start, "[s]")




