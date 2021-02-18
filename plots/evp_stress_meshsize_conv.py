from seaice import *
from firedrake import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "./output/evp_stress_meshsize_conv"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 1
dumpfreq = 10
timescale = 10

dirname = path + "/test.pvd"
plot_dirname = path + "/evp_stress_energy_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = np.arange(3, 100, 1)
length = 5 * 10 ** 5
bcs_values = [0, 1, 1]
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
    ics_values = [0, x / length, as_matrix([[0, 0], [0, 0]])]
    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
    forcing = [ocean_curr]
    evps = ElasticViscousPlasticStress(mesh=mesh, length=length, bcs_values=bcs_values, ics_values=ics_values,
                                       timestepping=timestepping, output=output, params=params, solver_params=solver,
                                       forcing=forcing, stabilised=False)

    t = 0

    while t < timescale - 0.5 * timestep:
        evps.solve(evps.usolver)
        evps.update(evps.u0, evps.u1)
        t += timestep
    energy_values.append(Energy.compute(evps.u1))

end = time()
print(end - start, "[s]")

energy_slope = format(np.polyfit(np.log(number_of_triangles), np.log(energy_values), 1)[0], '.3f')

plt.loglog(number_of_triangles, energy_values, label='Gradient = {}'.format(energy_slope))
plt.title('EVP (Matrix Solve) Energy vs. Meshsize')
plt.legend(loc='best')
plt.xlabel('Meshsize')
plt.ylabel('Energy of Solution')
plt.savefig(plot_dirname)
