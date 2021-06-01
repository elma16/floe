import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/srt-mesh"
Path(path).mkdir(parents=True, exist_ok=True)

"""
TEST 1 : STRAIN RATE TENSOR

Sheared square mesh
"""

zero = Constant(0)
zero_vector = Constant(as_vector([0, 0]))


if "--test" in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10 ** 5
    timescale = 10
else:
    timestep = 1
    dumpfreq = 1
    timescale = 10

dirname = path + "/u_timescale={}_timestep={}_CR_mesh_new.pvd".format(
    timescale, timestep
)
title = "Test Plot"
diagnostic_dirname = path + "/strain_rate_T={}_t={}.nc".format(timescale, timestep)
plot_dirname = path + "/strain_rate_error_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length, "y")
Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x + 0.5 * y, y]))
mesh.coordinates.assign(f)

x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

ic = {"u": v_exp, "a": 1, "h": 1}
conditions = Conditions(ic=ic)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = ViscousPlastic(
    mesh=mesh,
    conditions=conditions,
    timestepping=timestepping,
    output=output,
    params=params,
    solver_params=solver,
)

zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
sigma = zeta * srt.strain(grad(srt.u1))
sigma_exp = zeta * srt.strain(grad(v_exp))

eqn = srt.momentum_equation(
    srt.h,
    srt.u1,
    srt.u0,
    srt.p,
    sigma,
    params.rho,
    zero_vector,
    conditions.ocean_curr,
    params.rho_a,
    params.C_a,
    params.rho_w,
    params.C_w,
    conditions.geo_wind,
    params.cor,
    timestep,
)
eqn += timestep * inner(div(sigma_exp), srt.p) * dx

srt.assemble(eqn, srt.u1, srt.bcs, solver.srt_params)

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
    print("Error:", Error.compute(srt.u1, w))
end = time()
print(end - start, "[s]")

plotter = Plotter(
    dataset_dirname=diagnostic_dirname,
    diagnostic="error",
    plot_dirname=plot_dirname,
    timestepping=timestepping,
    title=title,
)

plotter.plot("semilogy")
