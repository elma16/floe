from seaice import *
from firedrake import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "./output/srt_meshsize_conv"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 1
dumpfreq = 10
timescale = 10

title = "Meshsize Convergence Plot"
dirname = "test.pvd"
plot_dirname = path + "/strain_rate_error_T={}_t={}_2-100-stabilised.png".format(timescale, timestep)

number_of_triangles = np.arange(2, 50, 1)
length = 5 * 10 ** 5
pi_x = pi / length
bcs_values = [0]
forcing = []
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()
energy_values = []
start = time()
for values in number_of_triangles:
    print(values)
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
    ics_values = [v_exp]
    srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, forcing=forcing, ics_values=ics_values,
                         timestepping=timestepping, output=output, params=params, solver_params=solver, stabilised=True)
    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep
    energy_values.append(Energy.compute(srt.u1))
end = time()
print(end - start, "[s]")

gradient = format((np.log(energy_values[0]) - np.log(energy_values[-1])) / (
            np.log(number_of_triangles[0]) - np.log(number_of_triangles[-1])), ".2f")

plt.loglog(number_of_triangles, energy_values, label="Timescale = {}".format(timescale))
plt.text(30, 4.6 * 10, 'Gradient = {}'.format(gradient), fontsize=12)
plt.ylabel(r'Energy of solution')
plt.xlabel(r'Meshsize')
plt.title(title)
plt.legend(loc='best')
plt.savefig(plot_dirname)
