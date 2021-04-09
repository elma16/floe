from seaice import *
from firedrake import *

timestep = 1
dumpfreq = 10**3
timescale = 10

dirname = "./output/test-output/u.pvd"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length)

x, y = SpatialCoordinate(mesh)
    
pi_x = pi / length

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

ic = {'u': 0, 'a' : x / length, 'h' : 0.5}
stabilised =  {'state': True , 'alpha': 1}
conditions = Conditions(family='CG',ocean_curr=ocean_curr,ic=ic,stabilised=stabilised)

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                     solver_params=solver)

t = 0

while t < timescale - 0.5 * timestep:
    srt.solve(srt.usolver)
    srt.update(srt.u0, srt.u1)
    t += timestep
    
def test_vp_model_compile():
    assert t > 0










