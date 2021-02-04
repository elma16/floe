import sys
from seaice import *
from firedrake import *
from time import time

# TEST 1 : STRAIN RATE TENSOR

"""
rho = 1 
h = 1
A = 1
No forcing
"""

if '--test' in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10 ** 5
    timescale = 10
else:
    timestep = 1
    dumpfreq = 10
    timescale = 10

dirname = "./output/strain_rate_tensor/u_timescale={}_timestep={}.pvd".format(timescale, timestep)
error_dirname = "./plots/strain_rate_error.png"
energy_dirname = "./plots/strain_rate_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
bcs_values = [0]
ics_values = [0]

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, ics_values=ics_values, timestepping=timestepping,
                     output=output, params=params, solver_params=solver)

t = 0
start = time()
while t < timescale - 0.5 * timestep:
    srt.solve(srt.usolver)
    srt.update(srt.u0, srt.u1)
    #srt.data['velocity'].append(Function(srt.u1))
    srt.dump(srt.u1, t)
    t += timestep
    srt.progress(t)
end = time()
print(end - start, "[s]")


