from seaice import *
from firedrake import *
from pathlib import Path

Path("./output/evp").mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP

seems to work up to a timestep of 12
'''

timestep = 0.1
dumpfreq = 10 ** 3
timescale = timestep * dumpfreq

stabilise = False
family = 'CR'

dirname = "./output/evp/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(timescale, timestep, stabilise,
                                                                                       family)
title = "EVP Plot"
diagnostic_dirname = "./output/evp/evp.nc"
plot_dirname = "./output/evp/evp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': [0, 1, 1],
              'ic': [0, x / length, as_matrix([[0, 0], [0, 0]])],
              'ocean_curr': ocean_curr}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

# using theta=1 for backward euler
evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                            params=params, solver_params=solver, stabilised=stabilise, family=family,theta=1)

diag = OutputDiagnostics(description="EVP Test", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    u0, s0 = evp.w0.split()
    print(norm(u0))
    evp.solve(evp.usolver)
    rel_error = Error.compute(evp.u1, u0) / norm(evp.u1)
    print('rel_error',rel_error)
    if rel_error < 10**(-10):
        print('time',t)
        break
    evp.update(evp.w0, evp.w1)
    # diag.dump(evp.u1, t)
    evp.dump(evp.u1, evp.s1, t=t)
    t += timestep
    evp.progress(t)

# plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=plot_dirname,
#                  timestepping=timestepping, title=title)

# plotter.plot()
