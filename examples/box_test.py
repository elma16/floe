import sys
from seaice import *
from firedrake import *
from pathlib import Path
Path("./output/bt").mkdir(parents=True, exist_ok=True)

# TEST 3 : BOX TEST

if '--test' in sys.argv:
    timestep = 600
    number_of_triangles = 71
    day = 60 * 60 * 24
    month = 31 * day
    timescale = month
    dumpfreq = 100
else:
    timescale = 100
    number_of_triangles = 30
    timestep = 1
    dumpfreq = 10

dirname = "./output/bt/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

plot_dirname = "./output/bt/box_test_energy.png"

length = 10 ** 6
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
t0 = Constant(0)
geo_wind = as_vector(
    [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
     5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])

bcs_values = [0, 1, 1]
ics_values = [0, 1, x / length]
forcing = [ocean_curr, geo_wind]

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

bt = ElasticViscousPlasticTransport(mesh=mesh, length=length, bcs_values=bcs_values, ics_values=ics_values,
                                    timestepping=timestepping, output=output, params=params, solver_params=solver,
                                    forcing=forcing)

t = 0
while t < timescale - 0.5 * timestep:
    bt.solve(bt.usolver)
    bt.update(bt.w0, bt.w1)
    bt.dump(bt.w1, t)
    t += timestep
    bt.progress(t)
