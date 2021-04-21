import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/srt"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 1 : STRAIN RATE TENSOR



'''

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

dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)
title = "Test Plot"
diagnostic_dirname = path + "/strain_rate_T={}_t={}.nc".format(timescale, timestep)
plot_dirname = path + "/strain_rate_error_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length)

x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

ic = {'u': v_exp, 'a' : 1, 'h' : 1}
stabilised = {'state':False, 'alpha':10}
conditions = Conditions(ic=ic, theta=1, stabilised=stabilised)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()

params = SeaIceParameters(rho=1,rho_a=zero,C_a=zero,rho_w=zero,C_w=zero,cor=zero)

srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                     solver_params=solver)

zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
sigma = zeta * srt.strain(grad(srt.uh))
sigma_exp = zeta * srt.strain(grad(conditions.ic['u']))

eqn = momentum_equation(srt.h, srt.u1, srt.u0, srt.p, sigma, params.rho, zero_vector, conditions.ocean_curr,
                        params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, timestep)
eqn += inner(div(sigma_exp), srt.p) * dx

srt.assemble(eqn,srt.u1,srt.bcs,solver.srt_params)

diag = OutputDiagnostics(description="test 1", dirname=diagnostic_dirname)

t = 0

w = Function(srt.V).interpolate(v_exp)
start = time()
while t < timescale - 0.5 * timestep:
    srt.solve(srt.usolver)
    srt.update(srt.u0, srt.u1)
    diag.dump(srt.u1, t, v_exp)
    srt.dump(srt.u1, w, t=t)
    t += timestep
    srt.progress(t)
    print(Error.compute(srt.u1, w))
end = time()
print(end - start, "[s]")

plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='error', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot('loglog')

