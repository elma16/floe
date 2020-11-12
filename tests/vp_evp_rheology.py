import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.parameters import *
from solvers.mevp_solver import *
from solvers.evp_solver import *
from solvers.forward_euler_solver import *

"""
from Mehlmann and Korn, 2020
Section 4.2
VP+EVP Test 1
Solve a modified momentum equation
L_x = L_y = L = 500000
vw_1 = 0.1*(2y-L_y)/L_y
vw_2 = -0.1*(L_x-2x)/L_x
v(0) = 0
h = 1
A = x/L_x
"""


def vp_evp_test_explicit(timescale=10, timestep=10 ** (-1), number_of_triangles=35, rheology="VP", advection=False,
                         solver="FE", stabilised=0, subcycle=100, output=False, last_frame=False):
    """
    Solving explicitly using the method in the paper
    """
    print(
        '\n******************************** {rheo} MODEL TEST ********************************\n'.format(rheo=rheology))
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CG", 1)

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

    # initial conditions

    u0.assign(0)
    u1.assign(u0)
    if not advection:
        h1 = Constant(1)
        a1.interpolate(x / L)
    if advection:
        h0.assign(1)
        h1.assign(h0)
        a0.interpolate(x / L)
        a1.assign(a0)

    if solver == "mEVP":
        beta = Constant(500)
    else:
        beta = Constant(1)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(u1) + transpose(grad(u1)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h1 * exp(-C * (1 - a1))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = 0.5 * P / Delta
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * P * Identity(2)

    if stabilised == 0:
        stab_term = 0
    if stabilised == 1:
        if rheology == "VP":
            # what does the paper mean by zeta_e?
            stab_term = 2 * a_vp * avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(u1), jump(v))) * dS
            if solver == "mEVP":
                stab_term = (a_mevp * avg(CellVolume(mesh)) * P) / (d * FacetArea(mesh)) * (dot(jump(u1), jump(v))) * dS
        elif rheology == "EVP":
            stab_term = (a_evp * avg(CellVolume(mesh)) * P) / (d * FacetArea(mesh)) * (dot(jump(u1), jump(v))) * dS
            if solver == "mEVP":
                stab_term = (a_mevp * avg(CellVolume(mesh)) * P) / (d * FacetArea(mesh)) * (dot(jump(u1), jump(v))) * dS

    # momentum equation (used irrespective of advection occurring or not)

    # diverges if i pick h1 -> h0
    lm = inner(beta * rho * h1 * (u1 - u0) + timestep * rho_w * C_w * sqrt(dot(u1 - ocean_curr, u1 - ocean_curr)) * (
            u1 - ocean_curr), v) * dx
    lm += timestep * inner(sigma, grad(v)) * dx
    lm += stab_term

    if advection:
        lh = timestep * inner(h1 - h0, w) * dx
        lh -= inner(u1 * h1, grad(w)) * dx
        la = timestep * inner(a1 - a0, w) * dx
        la -= inner(u1 * a1, grad(w)) * dx

    t = 0.0
    bcs = [DirichletBC(V, 0, "on_boundary")]

    if not advection:
        if rheology == "VP" and solver == "FE":
            all_u, all_h, all_a = forward_euler_solver(u1, u0, lm, bcs, t, timestep, timescale, output)
        elif rheology == "VP" and solver == "mEVP":
            all_u, all_h, all_a = mevp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T,
                                              timescale, output)
        elif rheology == "EVP" and solver == "EVP":
            all_u, all_h, all_a = evp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T,
                                             timescale, output)
        elif rheology == "EVP" and solver == "mEVP":
            all_u, all_h, all_a = mevp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T,
                                              timescale, output)
    if advection:
        if rheology == "VP" and solver == "FE":
            all_u, all_h, all_a = forward_euler_solver(u1, u0, lm, bcs, t, timestep, timescale,
                                                       output, advection, lh, la, h1, h0, a1, a0)
        elif rheology == "VP" and solver == "mEVP":
            all_u, all_h, all_a = mevp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta,
                                              timescale, output, advection, lh, la, h1, h0, a1, a0)
        elif rheology == "EVP" and solver == "EVP":
            all_u, all_h, all_a = evp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta,
                                             timescale, output, advection, lh, la, h1, h0, a1, a0)
        elif rheology == "EVP" and solver == "mEVP":
            all_u, all_h, all_a = mevp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta,
                                              timescale, output, advection, lh, la, h1, h0, a1, a0)

    print('...done!')

    return all_u, all_h, all_a, mesh, zeta


def evp_test_implicit(timescale=10, timestep=10 ** (-1), number_of_triangles=35, output=False, last_frame=False):
    """
    Solving using an implicit midpoint method and mixed function spaces.
    """
    print('\n******************************** IMPLICIT EVP MODEL TEST ********************************\n')

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    S = TensorFunctionSpace(mesh, "DG", 0)
    U = FunctionSpace(mesh, "CR", 1)
    W = MixedFunctionSpace([V, S])

    a = Function(U)

    w0 = Function(W)

    u0, s0 = w0.split()

    x, y = SpatialCoordinate(mesh)

    p, q = TestFunctions(W)

    # initial conditions

    u0.assign(0)
    a.interpolate(x / L)
    h = Constant(1)

    # ep_dot = 0.5 * (grad(u0) + transpose(grad(u0)))
    # ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)
    # P = P_star * h * exp(-C * (1 - a))
    # Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)
    # zeta = 0.5 * P / Delta
    # eta = zeta * e ** (-2)

    # s0.interpolate(2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * P * Identity(2))

    s0.assign(as_matrix([[1, 2], [3, 4]]))

    w1 = Function(W)
    w1.assign(w0)
    u1, s1 = split(w1)
    u0, s0 = split(w0)

    uh = 0.5 * (u0 + u1)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = 0.5 * P / Delta

    sh = 0.5 * (s1 + s0)

    # constructing the equations used

    lm = inner(p, rho * h * (u1 - u0)) * dx
    lm += timestep * inner(grad(p), sh) * dx
    lm -= timestep * inner(p, C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr)) * dx(degree=3)
    lm += inner(q, (s1 - s0) + timestep * (
            e ** 2 / (2 * T) * sh + ((1 - e ** 2) / (4 * T) * tr(sh) + P / (4 * T)) * Identity(2))) * dx
    lm -= inner(q * zeta * timestep / T, ep_dot) * dx

    bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]
    params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu", 'mat_type': 'aij'}
    uprob = NonlinearVariationalProblem(lm, w1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    u1, s1 = w1.split()

    # writing a pathname which depends on the variables chosen
    pathname = "./output/evp_alt/u_T={}_k={}_N={}.pvd".format(timescale, timestep, number_of_triangles)

    t = 0.0
    all_u = []

    if output and last_frame:
        ufile = File(pathname)
        ufile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            all_u.append(Function(u1))

        print('... EVP problem solved...\n')
        ufile.write(u1, time=t)

    elif output and not last_frame:
        ufile = File(pathname)
        ufile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            ufile.write(u1, time=t)
            all_u.append(Function(u1))
        print('... EVP problem solved...\n')
    else:
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        print('... EVP problem solved...\n')
    print('...done!')
    return all_u


def evp_test_implicit_matrix(timescale=10, timestep=10 ** (-1), number_of_triangles=35, output=False, last_frame=False):
    """
    Solving test 2 using the implicit midpoint rule, but solving a matrix system rather than using a mixed function space.

    Solution Strategy:

    Apply the implicit midpoint rule to the coupled system of PDEs.
    Solve sigma^{n+1} in terms of sigma^{n},v^{n},v^{n+1}.
    Plug in sigma^{n+1} into the momentum equation and solve exactly for v^{n+1}.
    """
    print('\n******************************** IMPLICIT EVP MODEL TEST ********************************\n')

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    S = TensorFunctionSpace(mesh, "DG", 0)
    U = FunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u0 = Function(V)
    u1 = Function(V)

    # stress tensors
    sigma0 = Function(S)
    sigma1 = Function(S)

    uh = 0.5 * (u1 + u0)

    a = Function(U)

    # test functions
    v = TestFunction(V)
    w = TestFunction(S)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u0.assign(0)
    h = Constant(1)
    a.interpolate(x / L)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = 0.5 * P / Delta
    eta = zeta * e ** (-2)

    # initalising the internal stress tensor
    sigma0.interpolate(2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * P * Identity(2))
    sigma1.assign(sigma0)

    # using the implicit midpoint rule formulation of the tensor equation, find sigma^{n+1} in terms of sigma^{n},v^{n+1},v^{n}
    def sigma_next(timestep, e, zeta, T, ep_dot, sigma, P):
        A = 1 + 0.25 * (timestep * e ** 2) / T
        B = timestep * 0.125 * (1 - e ** 2) / T
        rhs = (1 - (timestep * e ** 2) / (4 * T)) * sigma - timestep / T * (
                0.125 * (1 - e ** 2) * tr(sigma) * Identity(2) - 0.25 * P * Identity(2) + zeta * ep_dot)
        C = (rhs[0, 0] - rhs[1, 1]) / A
        D = (rhs[0, 0] + rhs[1, 1]) / (A + 2 * B)
        sigma00 = 0.5 * (C + D)
        sigma11 = 0.5 * (D - C)
        sigma01 = rhs[0, 1]
        sigma = as_matrix([[sigma00, sigma01], [sigma01, sigma11]])

        return sigma

    # momentum equation (used irrespective of advection occurring or not)

    s = sigma_next(timestep, e, zeta, T, ep_dot, sigma0, P)

    sh = 0.5 * (s + sigma0)

    lm = inner(rho * h * (u1 - u0), v) * dx + timestep * inner(
        rho_w * C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr), v) * dx(degree=3)
    lm += timestep * inner(sh, grad(v)) * dx

    ls = (inner(w, sigma1 - s)) * dx

    t = 0.0
    bcs = [DirichletBC(V, 0, "on_boundary")]

    pathname = './output/implicit_evp/u.pvd'

    if output and last_frame:
        outfile = File(pathname)
        outfile.write(u0, time=t)

        print('******************************** Implicit EVP Solver ********************************\n')
        while t <= timescale:
            solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
            solve(ls == 0, sigma1, solver_parameters=params)
            u0.assign(u1)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        outfile.write(u0, time=t)
        print('... EVP problem solved...\n')
    elif output and not last_frame:
        outfile = File(pathname)
        outfile.write(u0, time=t)

        print('******************************** Implicit EVP Solver ********************************\n')
        while t <= timescale:
            solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
            solve(ls == 0, sigma1, solver_parameters=params)
            u0.assign(u1)
            t += timestep
            outfile.write(u0, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')
    else:
        print('******************************** Implicit EVP Solver (NO OUTPUT) ********************************\n')
        while t <= timescale:
            solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
            solve(ls == 0, sigma1, solver_parameters=params)
            u0.assign(u1)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')

    print('...done!')
