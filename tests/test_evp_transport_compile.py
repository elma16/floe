import pytest
from seaice import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate, as_vector, pi)

# this one takes a while to run - could take out some of these options?

@pytest.mark.parametrize('family, theta',
                         [(b,c)
                          for b in ['CR', 'CG']
                          for c in [0,1/2,1]])


def test_evp_transport_model_compile(family, theta):
    timestep = 1
    dumpfreq = 10**3
    timescale = 1

    dirname = "./output/test-output/u.pvd"

    number_of_triangles = 35
    length = 5 * 10 ** 5
    mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length)

    x, y = SpatialCoordinate(mesh)

    pi_x = pi / length

    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

    ic = {'u': 0, 'a' : x / length, 'h' : 1, 's':as_vector([[0,0],[0,0]])}
    stabilised =  {'state': False , 'alpha': 1}
    advect = {'h': True,'a': True}
    conditions = Conditions(family=family,ocean_curr=ocean_curr,ic=ic,stabilised=stabilised,advect=advect,theta=theta)

    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()

    evp_transport = ElasticViscousPlasticTransport(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                                                   solver_params=solver)

    t = 0
    
    while t < timescale - 0.5 * timestep:
        evp_transport.solve(evp_transport.usolver)
        evp_transport.update(evp_transport.w0, evp_transport.w1)
        t += timestep

        
    assert t > 0
