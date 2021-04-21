from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp-stabilised"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP 

Find the value of the stabilisation parameter to obtain a stabilised velocity plot
'''

timestep = 0.1
dumpfreq = 10 ** 4
timescale = timestep * dumpfreq

for alpha in [1100,1200,1300,1400,1500,2000,2500,5000]:
                                                                            
    number_of_triangles = 35
    length = 5 * 10 ** 5
    mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
    x, y = SpatialCoordinate(mesh)

    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

    ic = {'u': 0, 'a' : x / length,'h':1, 's' : as_matrix([[0, 0], [0, 0]])}
    stabilised = {'state':True,'alpha':alpha}
    conditions = Conditions(ocean_curr=ocean_curr, ic=ic, stabilised=stabilised)

    dirname = path + "/u_timescale={}_timestep={}_stabilised={}_value={}.pvd".format(timescale, timestep, conditions.stabilised['state'],conditions.stabilised['alpha'])
    
    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()


    evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params, solver_params=solver)

    t = 0

    while t < timescale - 0.5 * timestep:
        u0, s0 = evp.w0.split()
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        evp.dump(evp.u1, evp.s1, t=t)
        t += timestep
        evp.progress(t)

