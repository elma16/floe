from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/gen_vp"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 10
dumpfreq = 10 ** 2
timescale = 10 ** 3

dirname = path + "test.pvd".format(timescale, timestep)
diagnostic_dirname = path + "test.nc".format(timescale, timestep)
plot_dirname = path + "/test_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': [0, 1, 1],
              'ic': [0, x / length],
              'ocean_curr': ocean_curr,
              'geo_wind': Constant(as_vector([0, 0]))}
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

vp = ViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                    params=params, solver_params=solver, stabilised=False, simple=False)

t = 0

start = time()
while t < timescale - 0.5 * timestep:
    vp.solve(vp.usolver)
    vp.update(vp.u0, vp.u1)
    vp.dump(vp.u1, t=t)
    t += timestep
    vp.progress(t)
end = time()
print(end - start, "[s]")
