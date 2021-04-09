import sys
from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/mk/bt-fixed"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 3 : BOX TEST
'''

if '--test' in sys.argv:
    timestep = 600
    number_of_triangles = 71
    day = 60 * 60 * 24
    month = 31 * day
    timescale = month
    dumpfreq = 144

    week = 7 * day
    timescale2 = week
    
else:
    number_of_triangles = 30
    timestep = 1
    dumpfreq = 1000
    timescale = timestep * dumpfreq
    timescale2 = timescale

plot_dirname = path + "/box_test_energy.png"

length = 10 ** 6
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
t0 = Constant(0)
geo_wind = as_vector(
    [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
     5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])

ic = {'u' : 0, 'h' : 1, 'a' : x / length, 's' : as_matrix([[0, 0], [0, 0]])}
conditions = Conditions(theta=0.5,geo_wind=geo_wind,ocean_curr=ocean_curr,ic=ic)
dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)
dirname_transport = path + "/u-trans_timescale={}_timestep={}.pvd".format(timescale, timestep)

timestepping_fixed = TimesteppingParameters(timescale=timescale, timestep=timestep)
timestepping_trans = TimesteppingParameters(timescale=timescale2, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
output_transport = OutputParameters(dirname=dirname_transport, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

bt = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping_fixed, output=output, params=params, solver_params=solver)

bt_transport = ElasticViscousPlasticTransport(mesh=mesh, conditions=conditions, timestepping=timestepping_trans, output=output_transport, params=params, solver_params=solver)


t = 0
while t < timescale - 0.5 * timestep:
    bt.solve(bt.usolver)
    bt.update(bt.w0, bt.w1)
    bt.dump(bt.u1, bt.s1, t=t)
    bt_transport.solve(bt_transport.usolver)
    bt_transport.update(bt_transport.w0, bt_transport.w1)
    bt_transport.dump(bt_transport.u1, bt_transport.a1, bt_transport.h1, t=t)
    t += timestep
    t0.assign(t)
    bt.progress(t)
