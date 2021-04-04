import sys
from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/bt"
Path(path).mkdir(parents=True, exist_ok=True)

# TEST 3 : BOX TEST

if '--test 1' in sys.argv:
    timestep = 600
    number_of_triangles = 71
    day = 60 * 60 * 24
    month = 31 * day
    timescale = month
    dumpfreq = 144
if '--test 2' in sys.argv:
    timestep = 600
    number_of_triangles = 71
    day = 60 * 60 * 24
    week = 7 * day
    timescale = week
    dumpfreq = 144    
else:
    number_of_triangles = 30
    timestep = 1
    dumpfreq = 1000
    timescale = timestep * dumpfreq

family = 'CG'

dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

plot_dirname = "./output/bt/box_test_energy.png"

length = 10 ** 6
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
t0 = Constant(0)
geo_wind = as_vector(
    [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
     5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])

conditions = {'bc': {'u' : 0},
              'ic': {'u' : 0, 'h' : 1,'a' : x / length},
              'ocean_curr': ocean_curr,
              'geo_wind': geo_wind}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

# TODO fix the divergence of the solution
bt = ElasticViscousPlasticTransport(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping,
                                    output=output, params=params, solver_params=solver, stabilised=False,family=family)

t = 0
while t < timescale - 0.5 * timestep:
    bt.solve(bt.usolver)
    bt.update(bt.w0, bt.w1)
    bt.dump(bt.u1, t=t)
    t += timestep
    t0.assign(t)
    bt.progress(t)
