from seaice import *
from firedrake import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "./output/srt_meshsize_conv"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 0.01
dumpfreq = 10**6
timescale = 10

dirname = path + "/test.pvd"
plot_dirname = path + "/strain_rate_error_T={}_t={}.png".format(timescale, timestep)
plot_dirname2 = path + "/strain_rate_energy_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = [5, 10, 20, 40]
length = 5 * 10 ** 5
pi_x = pi / length
bcs_values = [0]
forcing = []
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()
energy_values = []
error_values = []
start = time()

for values in number_of_triangles:
    print(values)
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
    ics_values = [v_exp]
    srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, forcing=forcing, ics_values=ics_values,
                         timestepping=timestepping, output=output, params=params, solver_params=solver,
                         stabilised=False, simple=True)

    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep
    energy_values.append(Energy.compute(srt.u1))
    error_values.append(Error.compute(srt.u1, v_exp))
end = time()
print(end - start, "[s]")

energy_slope = format(np.polyfit(np.log(number_of_triangles), np.log(energy_values), 1)[0], '.3f')
error_slope = format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f')

plt.figure(1)
plt.loglog(number_of_triangles, energy_values, label='Gradient = {}'.format(energy_slope))
plt.title('SRT Energy vs. Meshsize')
plt.legend(loc='best')
plt.xlabel('Meshsize')
plt.ylabel('Energy of Solution')
plt.savefig(plot_dirname2)
plt.figure(2)
plt.loglog(number_of_triangles, error_values, 'tab:orange', label='Gradient = {}'.format(error_slope))
plt.title('SRT Error vs. Meshsize')
plt.legend(loc='best')
plt.xlabel('Meshsize')
plt.ylabel('Error of solution')
plt.savefig(plot_dirname)
