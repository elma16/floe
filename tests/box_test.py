from firedrake import *
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

def box_test(number_of_triangles = 71):
    '''
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

    '''

    n = 30
    L = 1000000
    mesh = SquareMesh(n, n, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    W = FunctionSpace(mesh, "CR", 1)
    U = MixedFunctionSpace((V, W, W))

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # mean height of sea ice
    h_ = Function(W, name="Height")
    h = Function(W, name="HeightNext")

    # sea ice concentration
    A_ = Function(W, name="Concentration")
    A = Function(W, name="ConcentrationNext")

    # test functions
    v = TestFunction(V)
    w = TestFunction(W)
    q = TestFunction(W)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u_.assign(0)

    h = Constant(1)

    A = x / L

    timestep = 1 / n

    T = 100

    N_evp = 500

    # defining the constants to be used in the sea ice momentum equation:

    # the sea ice density
    rho = Constant(900)

    # Coriolis parameter
    cor = Constant(1.46 * 10 ** (-4))

    # air density
    rho_a = Constant(1.3)

    # air drag coefficient
    C_a = Constant(1.2 * 10 ** (-3))

    # water density
    rho_w = Constant(1026)

    # water drag coefficient
    C_w = Constant(5.5 * 10 ** (-3))

    # ice strength parameter
    P_star = Constant(27.5 * 10 ** 3)

    # ice concentration parameter
    C = Constant(20)

    #  ellipse ratio
    e = Constant(2)

    # geostrophic wind

    geo_wind = as_vector([5 + (sin(2 * pi * t / T) - 3) * sin(2 * pi * x / L) * sin(2 * pi * y / L),
                          5 + (sin(2 * pi * t / T) - 3) * sin(2 * pi * y / L) * sin(2 * pi * x / L)])

    # ocean current

    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # mEVP rheology

    alpha = Constant(500)
    beta = Constant(500)

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 1 / 2 * (grad(u) + transpose(grad(u)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - A))

    Delta_min = Constant(2 * 10 ** (-9))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)

    # solve the discretised sea ice momentum equation

    # constructing the discretised weak form

    # momentum equation
    # L_evp = (beta*rho*h/k_s*)

    Lm = (inner(rho * h * (u - u_) / timestep - rho * h * cor * as_vector([u[1] - ocean_curr[1], ocean_curr[0] - u[0]])
                + rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * dot(u - ocean_curr,
                                                                                       u - ocean_curr) * (
                        ocean_curr - u), v) +
          inner(sigma, grad(v))) * dx

    t = 0.0

    hfile = File('h.pvd')
    hfile.write(h_, time=t)
    all_hs = []
    end = T
    while (t <= end):
        solve(Lm == 0, u)
        u_.assign(u)
        t += timestep
        hfile.write(h_, time=t)
        print(t)

    try:
        fig, axes = plt.subplots()
        plot(all_hs[-1], axes=axes)
    except Exception as e:
        warning("Cannot plot figure. Error msg: '%s'" % e)

    try:
        plt.show()
    except Exception as e:
        warning("Cannot show figure. Error msg: '%s'" % e)
