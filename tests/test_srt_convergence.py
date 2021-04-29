from seaice import *
from firedrake import *
import numpy as np
import pytest

@pytest.mark.parametrize('state, theta',
                         [(a,c)
                          for a in [True, False]
                          for c in [0,1/2,1]])

def test_srt_initial_value(state, theta):
    timestep = 1
    dumpfreq = 10 ** 6
    timescale = 10

    zero = Constant(0)
    zero_vector = Constant(as_vector([0, 0]))

    dirname = "./output/test-output/test.pvd"
    plot_dirname = "./output/test-output/srt-conv.png"

    number_of_triangles = [5, 10, 20, 40, 100]

    error_values = []

    length = 5 * 10 ** 5
    pi_x = pi / length
    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters(rho=1, rho_a=zero, C_a=zero, rho_w=zero, C_w=zero, cor=zero)

    for values in number_of_triangles:
        mesh = SquareMesh(values, values, length)
        x, y = SpatialCoordinate(mesh)
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

        ic = {'u': v_exp, 'a' : 1, 'h' : 1}
        stabilised={'state':state ,'alpha':1}
        conditions = Conditions(ic=ic, theta=theta, stabilised=stabilised)
        srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                             solver_params=solver)

        zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
        sigma = zeta * srt.strain(grad(srt.u1))
        sigma_exp = zeta * srt.strain(grad(v_exp))

        eqn = srt.momentum_equation(srt.h, srt.u1, srt.u0, srt.p, sigma, params.rho, zero_vector, conditions.ocean_curr,
                                    params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, timestep)
        eqn += timestep * inner(div(sigma_exp), srt.p) * dx

        srt.assemble(eqn,srt.u1,srt.bcs,solver.srt_params)

        t = 0

        while t < timescale - 0.5 * timestep:
            srt.solve(srt.usolver)
            srt.update(srt.u0, srt.u1)
            t += timestep
            
        error_values.append(Error.compute(srt.u1, v_exp))

    error_slope = float(format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f'))


    assert round(error_slope + 2, 2) == 0

    
if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)





