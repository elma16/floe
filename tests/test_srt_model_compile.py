from seaice import *
from firedrake import *

timestep = 1
dumpfreq = 1
timescale = 10

dirname = "./output/test-output/u.pvd"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length)

x, y = SpatialCoordinate(mesh)

pi_x = pi / length
v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

ic = {'u': v_exp, 'a' : 1, 'h' : 1}
conditions = Conditions(ic=ic, steady_state=True)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
zero = Constant(0)
params = SeaIceParameters(rho=1,rho_a=zero,C_a=zero,rho_w=zero,C_w=zero,cor=zero)

srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                     solver_params=solver)

t = 0

while t < timescale - 0.5 * timestep:
    srt.solve(srt.usolver)
    srt.update(srt.u0, srt.u1)
    t += timestep


def test_srt_model_compile():
    assert t > 0






