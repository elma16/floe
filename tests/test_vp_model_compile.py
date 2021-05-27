import pytest
from seaice import *
from firedrake import *


@pytest.mark.parametrize('state, family, theta',
                         [(a, b, c)
                          for a in [True, False]
                          for b in ['CR', 'CG']
                          for c in [0, 1/2, 1]])


def test_vp_model_compile(state, family, theta):
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
    stabilised =  {'state': state , 'alpha': 1}
    conditions = Conditions(family=family, ocean_curr=ocean_curr, ic=ic, stabilised=stabilised, theta=theta)

    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()

    vp = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                        solver_params=solver)

    vp.assemble(vp.eqn ,vp.u1, vp.bcs, solver.srt_params)

    t = 0

    while t < timescale - 0.5 * timestep:
        vp.solve(vp.usolver)
        vp.update(vp.u0, vp.u1)
        t += timestep
    

    assert t > 0


@pytest.mark.parametrize('state, theta, order',
                         [(a, b, c)
                          for a in [True, False]
                          for b in [0, 1/2, 1]
                          for c in [0, 1]])


def test_vp_model_compile(state, theta, order):
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
    stabilised =  {'state': state , 'alpha': 1}
    conditions = Conditions(family='CG', ocean_curr=ocean_curr, ic=ic, stabilised=stabilised, theta=theta, order=order)

    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()

    vp = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                        solver_params=solver)

    vp.assemble(vp.eqn ,vp.u1, vp.bcs, solver.srt_params)

    t = 0

    while t < timescale - 0.5 * timestep:
        vp.solve(vp.usolver)
        vp.update(vp.u0, vp.u1)
        t += timestep
    

    assert t > 0






