from seaice import *
from firedrake import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "./output/bt_meshsize_conv"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 0.01
dumpfreq = 10
timescale = 1

dirname = path + "/test.pvd"
plot_dirname = path + "/bt_energy_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = np.arange(3, 20, 1)
length = 10 ** 6
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
    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
    t0 = Constant(0)
    geo_wind = as_vector(
        [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
         5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])
    ics_values = [0, 1, x / length]
    forcing = [ocean_curr, geo_wind]

    bt = ElasticViscousPlasticTransport(mesh=mesh, length=length, bcs_values=bcs_values, ics_values=ics_values,
                                        timestepping=timestepping, output=output, params=params, solver_params=solver,
                                        forcing=forcing, stabilised=False)

    t = 0

    while t < timescale - 0.5 * timestep:
        bt.solve(bt.usolver)
        bt.update(bt.w0, bt.w1)
        t += timestep
        t0.assign(t)
    u1, h1, a1 = bt.w1.split()
    energy_values.append(Energy.compute(u1))

end = time()
print(end - start, "[s]")

energy_slope = format(np.polyfit(np.log(number_of_triangles), np.log(energy_values), 1)[0], '.3f')

plt.loglog(number_of_triangles, energy_values, label='Gradient = {}'.format(energy_slope))
plt.title('Box Test Energy vs. Meshsize')
plt.legend(loc='best')
plt.xlabel('Meshsize')
plt.ylabel('Energy of Solution')
plt.savefig(plot_dirname)
