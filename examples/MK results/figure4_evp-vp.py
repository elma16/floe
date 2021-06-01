from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/mk/figure4"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP

Figure 4
'''

timestep = 0.1
dumpfreq = 10 
timescale = timestep * dumpfreq

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])


for state in [True, False]:
        
    ic = {'u': 0, 'a': x / length, 'h': 1,  's': as_matrix([[0, 0], [0, 0]])}
    stabilised = {'state': state , 'alpha': 1}
    conditions = Conditions(theta=0.5, ocean_curr=ocean_curr,ic=ic)
    dirname_evp = path + "/evp_u_timescale={}_timestep={}_stabilised={}.pvd".format(timescale, timestep, state)
    dirname_vp = path + "/vp_u_timescale={}_timestep={}_stabilised={}.pvd".format(timescale, timestep, state)
    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output_evp = OutputParameters(dirname=dirname_evp, dumpfreq=dumpfreq)
    output_vp = OutputParameters(dirname=dirname_vp, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()

    evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output_evp, params=params, solver_params=solver)

    vp = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output_vp, params=params, solver_params=solver)

    vp.assemble(vp.eqn, vp.u1, vp.bcs, solver.srt_params)

    t = 0

    while t < timescale - 0.5 * timestep:
        u0, s0 = evp.w0.split()
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        evp.dump(evp.u1, evp.s1, t=t)
        vp.solve(vp.usolver)
        vp.update(vp.u0, vp.u1)
        vp.dump(vp.u1, t=t)
        t += timestep
        evp.progress(t)
        





