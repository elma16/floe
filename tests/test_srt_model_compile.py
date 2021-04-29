import pytest
from seaice import *
from firedrake import *


@pytest.mark.parametrize('state, family, theta',
                         [(a,b,c)
                          for a in [True, False]
                          for b in ['CR', 'CG']
                          for c in [0,1/2,1]])


def test_srt_model_compile(state, family, theta):
    timestep = 1
    dumpfreq = 10**6
    timescale = 10

    zero = Constant(0)
    zero_vector = Constant(as_vector([0, 0]))

    dirname = "./output/test-output/u.pvd"
    number_of_triangles = 35
    length = 5 * 10 ** 5
    mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, length)

    x, y = SpatialCoordinate(mesh)

    pi_x = pi / length
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    ic = {'u': v_exp, 'a' : 1, 'h' : 1}
    stabilised={'state':state ,'alpha':1}
    conditions = Conditions(ic=ic, family=family, theta=theta)
    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    zero = Constant(0)
    params = SeaIceParameters(rho=1, rho_a=zero, C_a=zero, rho_w=zero, C_w=zero,cor=zero)

    srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                         solver_params=solver)

    zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
    sigma = zeta * srt.strain(grad(srt.u1))
    sigma_exp = zeta * srt.strain(grad(conditions.ic['u']))

    eqn = momentum_equation(srt.h, srt.u1, srt.u0, srt.p, sigma, params.rho, zero_vector, conditions.ocean_curr,
                            params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, timestep)
    eqn -= inner(div(sigma_exp), srt.p) * dx

    srt.assemble(eqn,srt.u1,srt.bcs,solver.srt_params)

    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep

    assert t > 0

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)






