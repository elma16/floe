import sys
from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/bt-adv"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 3 : BOX TEST

--test : one week of advection
'''

if '--test' in sys.argv:
    timestep = 600
    number_of_triangles = 30
    day = 60 * 60 * 24
    week = 7 * day
    timescale = week
    dumpfreq = 3000
    
else:
    number_of_triangles = 30
    timestep = 1
    dumpfreq = 1000
    timescale = timestep * dumpfreq

dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)
title = "EVP Fixed Energy Plot"
diagnostic_dirname = path + "/box_test_energy_T={}_t={}.nc".format(timescale, timestep)
plot_dirname = path + "/EVP_box_test_energy_T={}_t={}.png".format(timescale, timestep)

length = 10 ** 6
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
t0 = Constant(0)
geo_wind = as_vector(
    [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
     5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])

ic  = {'u' : 0, 'h' : 1, 'a' : x / length, 's' : as_matrix([[0, 0], [0, 0]])}
conditions = Conditions(theta=0.5,family='CG',geo_wind=geo_wind,ocean_curr=ocean_curr,ic=ic)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

bt = ElasticViscousPlasticTransport(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                                    solver_params=solver)

diag = OutputDiagnostics(description="test 3", dirname=diagnostic_dirname)

t = 0
while t < timescale - 0.5 * timestep:
    bt.solve(bt.usolver)
    bt.update(bt.w0, bt.w1)
    diag.dump(bt.w1, t=t)
    bt.dump(bt.u1, bt.a1, bt.h1, t=t)
    t += timestep
    t0.assign(t)
    bt.progress(t)

    
plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='error', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot('semilogy')
