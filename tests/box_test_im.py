import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.parameters import *


def box_test_im(timescale=2678400, timestep=600, number_of_triangles=71, subcycle=500, advection=False,
                output=False, stabilisation=0):
    """
    from Mehlmann and Korn, 2020
    Section 4.3
    Box-Test conditions
    Domain:
        L_x = L_y = 1000000 (meters)
    ocean current:
        o_1 = 0.1*(2*y - L_y)/L_y
        o_2 = -0.1*(L_x - 2*x)/L_x
    wind velocity:
        v_1 = 5 + sin(2*pi*t/T)-3)*(sin(2*pi*x/L_x)*sin(2*pi*y/L_y)
        v_2 = 5 + sin(2*pi*t/T)-3)*(sin(2*pi*y/L_x)*sin(2*pi*x/L_y)
    timestep:
        k = 600 (seconds)
    subcycles:
        N_evp = 500
    total time:
        one month T = 2678400 (seconds)
    Initial Conditions:
        v(0) = 0
        h(0) = 1
        A(0) = x/L_x

    Solved using the mEVP solver

    number_of_triangles : for the paper's 15190 edges, between 70 and 71 are required

    """
    print('\n******************************** BOX TEST ********************************\n')

    L = 1000000

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CR", 1)
    W = MixedFunctionSpace([V, U, U])

    w0 = Function(W)

    u0, h0, a0 = w0.split()

    # test functions

    p, q, r = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u0.assign(0)
    h0.assign(1)
    a0.interpolate(x / L)

    w1 = Function(W)
    w1.assign(w0)

    u0, h0, a0 = split(w0)

    u1, h1, a1 = split(w1)

    uh = 0.5 * (u0 + u1)
    hh = 0.5 * (h0 + h1)
    ah = 0.5 * (a0 + a1)
    # ocean current

    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * hh * exp(-C * (1 - ah))

    Delta_min = Constant(2 * 10 ** (-9))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

    # initalise geo_wind
    t = 0.0

    geo_wind = as_vector([5 + (sin(2 * pi * t / timescale) - 3) * sin(2 * pi * x / L) * sin(2 * pi * y / L),
                          5 + (sin(2 * pi * t / timescale) - 3) * sin(2 * pi * y / L) * sin(2 * pi * x / L)])

    lm = inner(rho * hh * (u1 - u0), p) * dx
    lm -= timestep * inner(rho * hh * cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0] - uh[0]]), p) * dx
    lm += timestep * inner(
        rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * dot(uh - ocean_curr, uh - ocean_curr) * (
                ocean_curr - uh), p) * dx
    lm += inner(sigma, grad(p)) * dx

    if advection:
        lm += inner((h1 - h0), q) * dx
        lm -= timestep * inner(uh * hh, grad(q)) * dx
        lm += inner((a1 - a0), r) * dx
        lm -= timestep * inner(uh * ah, grad(r)) * dx

    bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]

    params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu", 'mat_type': 'aij'}

    uprob = NonlinearVariationalProblem(lm, w1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    u1, h1, a1 = w1.split()

    ufile = File('./output/box_test/box_test_alt.pvd')
    t = 0.0
    ufile.write(u1, time=t)
    all_u = []

    ndump = 10
    dumpn = 0

    while t < timescale - 0.5 * timestep:
        t += timestep

        usolver.solve()
        w0.assign(w1)

        dumpn += 1
        print("Time:", t, "[s]")
        print(int(min(t / timescale * 100, 100)), "% complete")
        if dumpn == ndump:
            dumpn -= ndump
            ufile.write(u1, time=t)
            all_u.append(Function(u1))

    print('...done!')


box_test_im(timescale=10, timestep=1, subcycle=5, advection=True)
