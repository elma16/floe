from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp-adaptive-timestepping"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 25
dumpfreq = 10 ** 6
timescale = 10 ** 5

title = "EVP Plot"
diagnostic_dirname = path + "/evp.nc"
plot_dirname = path + "/evp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)
c = Constant(0.1)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
sigma = as_matrix(
    [
        [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
        [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
    ]
)

ocean_curr = as_vector(
    [0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length]
)
ocean_curr = c * ocean_curr

ic = {"u": 0, "a": x / length, "h": 1, "s": as_matrix([[0, 0], [0, 0]])}

stabilised = {"state": False, "alpha": 1}

conditions = Conditions(
    ic=ic, ocean_curr=ocean_curr, stabilised=stabilised, family="CG"
)

dirname = path + "/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(
    timescale, timestep, conditions.stabilised["state"], conditions.family
)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()
evp = ElasticViscousPlastic(
    mesh=mesh,
    conditions=conditions,
    timestepping=timestepping,
    output=output,
    params=params,
    solver_params=solver,
)

evp.assemble(evp.eqn, evp.w1, evp.bcs, solver.srt_params)

evp.u1, evp.s1 = evp.w1.split()

diag = OutputDiagnostics(description="test 1", dirname=diagnostic_dirname)

t = 0

d = Function(evp.D)

while t < timescale - 0.5 * timestep:
    u0, s0 = evp.w0.split()
    evp.solve(evp.usolver)
    evp.update(evp.w0, evp.w1)
    diag.dump(evp.w1, t=t)
    d.interpolate(evp.delta(evp.u1))
    evp.dump(evp.u1, evp.s1, d, t=t)
    t += timestep
    c.assign(min_value(1, c + timestep ** (-3)))
    # c.assign(conditional(lt(t,timescale/10),0.1,1))
    evp.progress(t)


plotter = Plotter(
    dataset_dirname=diagnostic_dirname,
    diagnostic="energy",
    plot_dirname=plot_dirname,
    timestepping=timestepping,
    title=title,
)

plotter.plot()
