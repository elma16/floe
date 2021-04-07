import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/figure3"
Path(path).mkdir(parents=True, exist_ok=True)

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10 ** 5
    timescale = 10
else:
    timestep = 1
    dumpfreq = 1
    timescale = 10

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
solver = SolverParameters()
params = SeaIceParameters()

for stab in [True,False]:
    conditions = {'bc': {'u': 0},
                  'ic': {'u': v_exp},
                  'stabilised':{'state':stab,'alpha':1},
                  'family':'CR',
                  'simple':True}

    dirname = path + "/u_timescale={}_timestep={}_family={}_stabilised={}.pvd".format(timescale, timestep, conditions['family'],conditions['stabilised']['state'])
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                         solver_params=solver)

    t = 0

    w = Function(srt.V).interpolate(v_exp)

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        srt.dump(srt.u1, w, t=t)
        t += timestep
        srt.progress(t)







