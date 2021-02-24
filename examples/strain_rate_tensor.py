import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path

path = "./output/srt"
Path(path).mkdir(parents=True, exist_ok=True)

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10 ** 5
    timescale = 10
else:
    timestep = 1
    dumpfreq = 1
    timescale = 100

dirname = path + "/u_timescale={}_timestep={}_new.pvd".format(timescale, timestep)
title = "Test Plot"
diagnostic_dirname = path + "/strain_rate_T={}_t={}.nc".format(timescale, timestep)
plot_dirname = path + "/strain_rate_error_T={}_t={}.png".format(timescale, timestep)

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)
pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
bcs_values = [0]
forcing = []
ics_values = [v_exp]

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = ViscousPlastic(mesh=mesh, length=length, bcs_values=bcs_values, forcing=forcing, ics_values=ics_values,
                     timestepping=timestepping, output=output, params=params, solver_params=solver,
                     stabilised=False, simple=True)

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
end = time()
print(end - start, "[s]")

plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='error', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot()
