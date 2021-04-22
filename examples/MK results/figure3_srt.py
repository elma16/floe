import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/mk/figure3"
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

zero = Constant(0)
zero_vector = Constant(as_vector([0, 0]))

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
solver = SolverParameters()
params = SeaIceParameters(rho=1, rho_a=zero, C_a=zero, rho_w=zero, C_w=zero, cor=zero)

for stab in [True,False]:
    ic = {'u':v_exp, 'a':1, 'h':1}
    stabilised = {'state':stab,'alpha':1}
    conditions = Conditions(ic=ic)
    
    dirname = path + "/u_timescale={}_timestep={}_family={}_stabilised={}.pvd".format(timescale, timestep, conditions.family,conditions.stabilised['state'])
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                         solver_params=solver)

    zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
    sigma = zeta * srt.strain(grad(srt.u1))
    sigma_exp = zeta * srt.strain(grad(v_exp))

    eqn = momentum_equation(srt.h, srt.u1, srt.u0, srt.p, sigma, params.rho, zero_vector, conditions.ocean_curr,
                            params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, timestep)
    eqn += timestep * inner(div(sigma_exp), srt.p) * dx

    srt.assemble(eqn, srt.u1, srt.bcs, solver.srt_params)

    t = 0

    w = Function(srt.V).interpolate(v_exp)

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        srt.dump(srt.u1, w, t=t)
        t += timestep
        srt.progress(t)







