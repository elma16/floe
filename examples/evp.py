from seaice import *
from firedrake import *
from pathlib import Path

Path("./output/evp").mkdir(parents=True, exist_ok=True)

# TEST 2 : EVP

# TODO velocities converge against stationary solution. Construct stationary solution

timestep = 10
dumpfreq = 10
timescale = 10 ** 3

dirname = "./output/evp/u_timescale={}_timestep={}_stabilised={}.pvd".format(timescale, timestep, False)
title = "EVP Plot"
diagnostic_dirname = "./output/evp/evp.nc"
plot_dirname = "./output/evp/evp_energy1000.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': [0, 1, 1], 'ic': [0, x / length, as_matrix([[1, 2], [3, 4]])], 'ocean_curr': ocean_curr}
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                            params=params, solver_params=solver, stabilised=False)

diag = OutputDiagnostics(description="EVP Test", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    evp.solve(evp.usolver)
    evp.update(evp.w0, evp.w1)
    diag.dump(evp.u1, t)
    evp.dump(evp.u1, t=t)
    t += timestep
    evp.progress(t)

plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot()
