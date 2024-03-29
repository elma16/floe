from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/vp"
Path(path).mkdir(parents=True, exist_ok=True)

"""
Viscous plastic with ocean current forcing, and with zeta = zeta_max
"""

timestep = 1
dumpfreq = 100
timescale = timestep * dumpfreq

title = "VP Plot"
diagnostic_dirname = path + "/vp.nc"
plot_dirname = path + "/vp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector(
    [0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length]
)

ic = {"u": 0, "a": x / length, "h": 1}
stabilised = {"state": True, "alpha": 1}
conditions = Conditions(
    family="CR", ocean_curr=ocean_curr, ic=ic, stabilised=stabilised, theta=0.5
)
dirname = path + "/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(
    timescale, timestep, conditions.stabilised["state"], conditions.family
)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

vp = ViscousPlastic(
    mesh=mesh,
    conditions=conditions,
    timestepping=timestepping,
    output=output,
    params=params,
    solver_params=solver,
)

diag = OutputDiagnostics(description="VP Test", dirname=diagnostic_dirname)

vp.assemble(vp.eqn, vp.u1, vp.bcs, solver.srt_params)

t = 0

while t < timescale - 0.5 * timestep:
    vp.solve(vp.usolver)
    vp.update(vp.u0, vp.u1)
    vp.dump(vp.u1, t=t)
    t += timestep
    vp.progress(t)
