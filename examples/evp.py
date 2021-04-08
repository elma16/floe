from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp2"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP
'''

timestep = 0.1
dumpfreq = 10 ** 4
timescale = timestep * dumpfreq

title = "EVP Plot"
diagnostic_dirname = path + "/evp.nc"
plot_dirname = path + "/evp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

ic =  {'u': 0, 'a': x/length, 'h': 1, 's': as_matrix([[0, 0], [0, 0]])}

conditions = Conditions(ic = ic, ocean_curr=ocean_curr)

dirname = path + "/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(timescale, timestep, conditions.stabilised['state'], conditions.family)
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                            solver_params=solver)

t = 0

while t < timescale - 0.5 * timestep:
    u0, s0 = evp.w0.split()
    evp.solve(evp.usolver)
    evp.update(evp.w0, evp.w1)
    evp.dump(evp.u1, evp.s1, t=t)
    t += timestep
    evp.progress(t)





