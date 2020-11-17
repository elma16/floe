import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.parameters import *
from solvers.mevp_solver import *
from solvers.solver_general import *

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


def box_test(timescale=2678400, timestep=600, number_of_triangles=71, subcycle=500, advection=False,
             output=False, stabilised=0):
    """solving the full system of coupled PDEs explicitly, in the method of the paper"""
    print('\n******************************** BOX TEST (mEVP solve) ********************************\n')
    L = 1000000

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u0 = Function(V)
    u1 = Function(V)

    # mean height of sea ice
    h0 = Function(U)
    h1 = Function(U)

    # sea ice concentration
    a0 = Function(U)
    a1 = Function(U)

    # test functions
    v = TestFunction(V)
    w = TestFunction(U)

    x, y = SpatialCoordinate(mesh)

    timestepc = Constant(timestep)

    # initial conditions

    u0.assign(0)
    u1.assign(u0)
    a0.interpolate(x / L)
    a1.assign(a0)
    if not advection:
        h1 = Constant(1)
    if advection:
        h0.assign(1)
        h1.assign(h0)

    # boundary condition
    h_in = Constant(0)
    a_in = Constant(0)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(u0) + transpose(grad(u0)))

    # ice strength
    P = P_star * h1 * exp(-C * (1 - a1))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

    if stabilised == 0:
        stab_term = 0
    elif stabilised == 1:
        stab_term = 2 * a_vp * avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(u1), jump(v))) * dS
    elif stabilised == 2:
        sigma = 0.5 * zeta * grad(u1)

    # initalise geo_wind
    t0 = Constant(0)

    geo_wind = as_vector([5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / L) * sin(2 * pi * y / L),
                          5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / L) * sin(2 * pi * x / L)])

    lm = inner(rho * h1 * (u1 - u0), v) * dx
    lm -= timestepc * inner(rho * h1 * cor * as_vector([u1[1] - ocean_curr[1], ocean_curr[0] - u1[0]]), v) * dx
    lm += timestepc * inner(
        rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * sqrt(dot(u1 - ocean_curr, u1 - ocean_curr)) * (
                ocean_curr - u1), v) * dx
    lm += timestepc * inner(sigma, grad(v)) * dx
    lm += stab_term

    # solving the transport equations using an upwind method
    if advection:
        dh_trial = TrialFunction(U)
        da_trial = TrialFunction(U)

        #LHS
        lhsh = w * dh_trial * dx
        lhsa = w * da_trial * dx

        n = FacetNormal(mesh)

        un = 0.5 * (dot(u1, n) + abs(dot(u1, n)))

        lh = timestepc * (h1 * div(w * u1) * dx
            - conditional(dot(u1, n) < 0, w * dot(u1, n) * h_in, 0.0) * ds
            - conditional(dot(u1, n) > 0, w * dot(u1, n) * h1, 0.0) * ds
            - (w('+') - w('-')) * (un('+') * a1('+') - un('-') * h1('-')) * dS)

        la = timestepc * (a1 * div(w * u1) * dx
            - conditional(dot(u1, n) < 0, w * dot(u1, n) * a_in, 0.0) * ds
            - conditional(dot(u1, n) > 0, w * dot(u1, n) * a1, 0.0) * ds
            - (w('+') - w('-')) * (un('+') * a1('+') - un('-') * a1('-')) * dS)

        hprob = LinearVariationalProblem(lhsh, lh, h1)
        hsolver = LinearVariationalSolver(hprob, solver_parameters=params)
        aprob = LinearVariationalProblem(lhsa, la, a1)
        asolver = LinearVariationalSolver(aprob, solver_parameters=params)

    bcs = [DirichletBC(V, 0, "on_boundary")]
    uprob = NonlinearVariationalProblem(lm, u1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    t = 0
    if advection:
        all_u = bt_solver(output, u0, u1, t, t0, timestep, timescale, usolver, sigma, ep_dot, P, zeta, subcycle,
                          advection, hsolver, h0, h1, asolver, a0, a1)
    elif not advection:
        all_u = bt_solver(output, u0, u1, t, t0, timestep, timescale, usolver, sigma, ep_dot, P, zeta, subcycle)

    print('...done!')

    return all_u


def box_test_im(timescale=2678400, timestep=600, number_of_triangles=71, output=False, stabilised=0, last_frame=False,
                advection = True):
    """solving the box test using the implicit midpoint rule"""
    print('\n******************************** BOX TEST ********************************\n')

    L = 1000000

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CR", 1)
    W = MixedFunctionSpace([V, U, U])

    w0 = Function(W)

    u0, h0, a0 = w0.split()

    #Delta = Function(U)

    # test functions

    p, q, r = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)

    timestepc = Constant(timestep)

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

    # ice strength
    P = P_star * hh * exp(-C * (1 - ah))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

    if stabilised == 0:
        stab_term = 0
    if stabilised == 1:
        stab_term = 2 * a_vp * avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(u1), jump(v))) * dS

    # viscosities
    zeta = 0.5 * P / Delta
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

    # initalise geo_wind
    t0 = Constant(0)

    geo_wind = as_vector([5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / L) * sin(2 * pi * y / L),
                          5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / L) * sin(2 * pi * x / L)])

    lm = inner(rho * hh * (u1 - u0), p) * dx
    lm -= timestepc * inner(rho * hh * cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0] - uh[0]]), p) * dx
    lm += timestepc * inner(
        rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (
                ocean_curr - uh), p) * dx
    lm += timestepc * inner(sigma, grad(p)) * dx
    lm += stab_term

    # adding the transport equations
    if advection:
        lm += inner(h1 - h0, q) * dx
        lm -= timestep * inner(uh * hh, grad(q)) * dx
        lm += inner(a1 - a0, r) * dx
        lm -= timestep * inner(uh * ah, grad(r)) * dx

    bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]

    params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu", 'mat_type': 'aij'}

    uprob = NonlinearVariationalProblem(lm, w1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    u1, h1, a1 = w1.split()

    all_u = []
    all_h = []
    all_a = []
    all_delta = []

    t = 0

    if output and last_frame:
        # just output the beginning and end frames
        ufile = File('./output/box_test/box_test_alt_u.pvd')
        hfile = File('./output/box_test/box_test_alt_h.pvd')
        afile = File('./output/box_test/box_test_alt_a.pvd')
        deltafile = File('./output/box_test/box_test_alt_delta.pvd')

        ufile.write(u1, time=t)
        hfile.write(h1, time=t)
        afile.write(a1, time=t)
        deltafile.write(Delta, time=t)

        while t < timescale - 0.5 * timestep:
            usolver.solve()
            w0.assign(w1)
            all_u.append(Function(u1))
            all_h.append(Function(h1))
            all_a.append(Function(a1))
            t += timestep
            t0.assign(t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        ufile.write(u1, time=t)
        hfile.write(h1, time=t)
        afile.write(a1, time=t)
        deltafile.write(Delta, time=t)

    elif output and not last_frame:
        # output the whole simulation
        ufile = File('./output/box_test/box_test_alt_u.pvd')
        hfile = File('./output/box_test/box_test_alt_h.pvd')
        afile = File('./output/box_test/box_test_alt_a.pvd')
        #deltafile = File('./output/box_test/box_test_alt_delta.pvd')

        ufile.write(u1, time=t)
        hfile.write(h1, time=t)
        afile.write(a1, time=t)
        #deltafile.write(Delta, time=t)

        while t < timescale - 0.5 * timestep:
            usolver.solve()
            w0.assign(w1)
            ufile.write(u1, time=t)
            hfile.write(h1, time=t)
            afile.write(a1, time=t)
            t += timestep
            t0.assign(t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            all_u.append(Function(u1))
            all_h.append(Function(h1))
            all_a.append(Function(a1))
    else:
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            w0.assign(w1)
            all_u.append(Function(u1))
            all_h.append(Function(h1))
            all_a.append(Function(a1))
            t += timestep
            t0.assign(t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

    print('...done!')

    return all_u, all_h, all_a, all_delta
