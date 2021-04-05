from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp_stress"
Path(path).mkdir(parents=True, exist_ok=True)

# TEST 2 : EVP

timestep = 20
dumpfreq = 10 ** 3
timescale = timestep * dumpfreq

dirname = path + "/u_timescale={}_timestep={}_stab={}.pvd".format(timescale, timestep, stabilise)
title = "EVP Plot"
diagnostic_dirname = path + "/evp.nc"
plot_dirname = path + "/evp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': {'u' : 0},
              'ic': {'u' : 0,'a' :  x / length},
              'ocean_curr': ocean_curr,
              'stabilise' : {'state':True, 'alpha':10},
              'family':'CR'}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

evps = ElasticViscousPlasticStress(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                                   solver_params=solver)

diag = OutputDiagnostics(description="EVP Matrix Test", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    evps.solve(evps.usolver, evps.ssolver)
    evps.update(evps.u0, evps.u1)
    diag.dump(evps.u1, t)
    evps.dump(evps.u1, t=t)
    t += timestep
    evps.progress(t)

plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot()
