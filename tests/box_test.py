import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from firedrake import *
from tests.parameters import *

def box_test(number_of_triangles = 71,timestep = 600,T=2678400,subcycle = 500):
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

    # geostrophic wind

    geo_wind = as_vector([5 + (sin(2 * pi * t / T) - 3) * sin(2 * pi * x / L) * sin(2 * pi * y / L),
                          5 + (sin(2 * pi * t / T) - 3) * sin(2 * pi * y / L) * sin(2 * pi * x / L)])

    # ocean current

    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

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

    outfile = File('./output/box_test/box-test.pvd')
    outfile.write(h_, time=t)
    all_hs = []
    end = T

    print('******************************** Forward solver ********************************\n')
    while t <= end:
        solve(Lm == 0, u)
        u_.assign(u)
        t += timestep
        outfile.write(h_, time=t)
        print(t)
    print('... forward problem solved...\n')

    print('...done!')

