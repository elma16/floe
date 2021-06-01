from seaice import *
from firedrake import (PeriodicSquareMesh, SpatialCoordinate, as_vector, pi, SquareMesh,
                       sin, as_matrix)
import numpy as np
import pytest

@pytest.mark.parametrize('state, norm_type, theta, family',
                         [(a, b, c, d)
                          for a in [True, False]
                          for b in ['L2', 'H1']
                          for c in [0,1/2,1]
                          for d in ['CR', 'CG']])


def test_evp_convergence(state, norm_type, theta, family):
    timestep = 1
    dumpfreq = 10 ** 6
    timescale = 2
    number_of_triangles = [5, 10, 20, 40, 80]
    length = 5 * 10 ** 5
    pi_x = pi / length

    zero = Constant(0)
    zero_vector = Constant(as_vector([0, 0]))

    dirname = "./output/test-output/test.pvd"
    plot_dirname = "./output/test-output/srt-conv.png"

    error_values = []

    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters(rho_a=zero, C_a=zero, cor=zero)

    for values in number_of_triangles:
        mesh = SquareMesh(values, values, length)
        x, y = SpatialCoordinate(mesh)

        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
        sigma_exp = as_matrix([[-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
                               [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)]])

        ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

        ic = {'u': v_exp, 'a' : 1, 'h' : 1, 's': sigma_exp}
        
        stabilised={'state':state ,'alpha':1}

        conditions = Conditions(ic=ic, ocean_curr=ocean_curr, stabilised=stabilised, family=family)

        evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                                    solver_params=solver)

        u1, s1 = split(evp.w1)
        u0, s0 = split(evp.w0)

        uh = (1-theta) * u0 + theta * u1
        sh = (1-theta) * s0 + theta * s1

        eqn = inner(params.rho * evp.h * (u1 - u0), evp.p) * dx
        eqn += timestep * inner(sh, grad(evp.p)) * dx
        eqn -= timestep * inner(params.rho_w * params.C_w * sqrt(dot(ocean_curr - uh, ocean_curr - uh)) * (ocean_curr - uh), evp.p) * dx

        # source terms in momentum equation=-==
        eqn += timestep * inner(div(sigma_exp), evp.p) * dx
        eqn += timestep * inner(params.rho_w * params.C_w * sqrt(dot(ocean_curr - v_exp, ocean_curr - v_exp)) * (ocean_curr - v_exp), evp.p) * dx

        zeta_exp = evp.zeta(evp.h, evp.a, evp.delta(v_exp))
        ep_dot_exp = evp.strain(grad(v_exp))
        rheology_exp = params.e ** 2 * sigma_exp + Identity(2) * 0.5 * ((1 - params.e ** 2) * tr(sigma_exp) + evp.Ice_Strength(evp.h, evp.a))
        zeta = evp.zeta(evp.h, evp.a, evp.delta(uh))
    
        eqn += inner(s1 - s0 + 0.5 * timestep * evp.rheology / params.T, evp.q) * dx
        eqn -= inner(evp.q * zeta * timestep / params.T, evp.ep_dot) * dx

        # source terms in rheology 
        eqn -= inner(0.5 * timestep * rheology_exp / params.T, evp.q) * dx
        eqn += inner(evp.q * zeta_exp * timestep / params.T, ep_dot_exp) * dx
    
        evp.assemble(eqn, evp.w1, evp.bcs, solver.srt_params)

        t = 0

        u1, s1 = evp.w1.split()
    
        evp.dump(u1, s1, t=0)

        while t < timescale - 0.5 * timestep:
            u0, s0 = evp.w0.split()
            evp.solve(evp.usolver)
            evp.update(evp.w0, evp.w1)
            t += timestep
            evp.dump(u1, s1, t=t)
            evp.progress(t)
            
        error_values.append(Error.compute(u1, v_exp, norm_type))
        
    h = [sqrt(2)*length/x for x in number_of_triangles]
    error_slope = float(format(np.polyfit(np.log(h), np.log(error_values), 1)[0], '.3f'))

    assert round(error_slope - 2, 1) == 0

    
if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)





