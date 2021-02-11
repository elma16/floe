import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path
Path("./output/srt").mkdir(parents=True, exist_ok=True)

# TEST 1 : STRAIN RATE TENSOR

if '--test' in sys.argv:
    timestep = 10 ** (-6)
    dumpfreq = 10 ** 5
    timescale = 10
else:
    timestep = 1
    dumpfreq = 10
    timescale = 10

dirname = "./output/srt/u_timescale={}_timestep={}.pvd".format(timescale, timestep)
title = "Test Plot"
diagnostic_dirname = "./output/srt/strain_rate.nc"
plot_dirname = "./output/srt/strain_rate_error.png"

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
                     timestepping=timestepping, output=output, params=params, solver_params=solver)

diag = OutputDiagnostics(description="test 1", dirname=diagnostic_dirname)

t = 0
start = time()
while t < timescale - 0.5 * timestep:
    srt.solve(srt.usolver)
    srt.update(srt.u0, srt.u1)
    diag.dump(srt.u1, v_exp, t)
    srt.dump(srt.u1, t)
    t += timestep
    srt.progress(t)
end = time()
print(end - start, "[s]")


plotter = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=plot_dirname,
                  timestepping=timestepping, title=title)

plotter.plot()

