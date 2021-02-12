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

title = "Test Plot"
dirname = "test.pvd"
plot_dirname = path + "/strain_rate_error_T={}.png".format(timescale, timestep)

number_of_triangles = np.arange(10, 200, 10)
length = 5 * 10 ** 5
pi_x = pi / length
bcs_values = [0]
forcing = []
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()
norm_values = []
start = time()
for values in number_of_triangles:
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
    ics_values = [v_exp]
    srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, forcing=forcing, ics_values=ics_values,
                         timestepping=timestepping, output=output, params=params, solver_params=solver)
    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep
    norm_values.append(norm(srt.u1, norm_type="H1"))
end = time()
print(end - start, "[s]")
plt.loglog(number_of_triangles, norm_values, label="Timescale = {}".format(timescale))
plt.ylabel(r'Energy of solution')
plt.xlabel(r'Meshsize')
plt.title(title)
plt.legend(loc='best')
plt.savefig(plot_dirname)

