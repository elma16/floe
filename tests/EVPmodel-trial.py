from firedrake import *
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")


def EVP_trial(T=10, timestep=10 ** (-1), number_of_triangles=30, subcycles=20):
    """
    from Mehlmann and Korn, 2020
    Section 4.2
    EVP Test 1
    Solve a modified momentum equation using the EVP solver, rather than the mEVP solver
    L_x = L_y = L = 500000
    vw_1 = 0.1*(2y-L_y)/L_y
    vw_2 = -0.1*(L_x-2x)/L_x
    v(0) = 0
    h = 1
    A = x/L_x
    """
    print('\n******************************** EVP TRIAL ********************************\n')
    n = number_of_triangles
    L = 500000
    mesh = SquareMesh(n, n, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    #
    A = Function(U)

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u_.assign(0)

    h = Constant(1)

    A.interpolate(x / L)

    # defining the constants to be used in the sea ice momentum equation:

    # the sea ice density
    rho = Constant(900)

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

    # ------------------------------------------------------------------------
    subcycle_timestep = timestep/subcycles

    #updating the EVP stress tensor
    sigma1 = sigma[0,0] + sigma [1,1]
    sigma2= sigma[0,0] - sigma[1,1]
    ep_dot1 = ep_dot[0,0] + ep_dot[1,1]
    ep_dot2 = ep_dot[0,0] - ep_dot[1,1]

    #subcycled update of EVP stress tensor

    sigma1 = (2 * subcycle_timestep * zeta * ep_dot1 - P / 2 + 2 * T * sigma1) / (2 * T + subcycle_timestep)
    sigma2 = (2 * subcycle_timestep * zeta * ep_dot2 - P / 2 + 2 * T * sigma2) / (
                2 * T + subcycle_timestep)

    #computing the entries of the stress tensor
    sigma[0, 1] = (2 * subcycle_timestep * zeta * ep_dot[0,1]+2*T*sigma[0,1])/(2*T+4*subcycles)
    sigma[0, 0] = (sigma1 + sigma2) / 2
    sigma[0, 0] = (sigma1 - sigma2) / 2

    #updating the mEVP stress tensor

    alpha = 500
    beta = 500
    sigma1 = 1 + (sigma1 + 2 * zeta * (ep_dot1 - P ) )/alpha
    sigma2 = 1 + (sigma2 * zeta * ep_dot2)/2*alpha
    sigma[0, 1] = 1 + (sigma[0,1] * zeta * ep_dot[0,1])/2*alpha
    #------------------------------------------------------------------------

    # momentum equation

    a = (inner(
        rho * h * (u - u_) / timestep + rho_w * C_w * sqrt(dot(u - ocean_curr, u - ocean_curr)) * (ocean_curr - u),
        v)) * dx
    a += inner(sigma, grad(v)) * dx

    t = 0.0

    outfile = File('./output/EVP_model/EVP_test.pvd')
    outfile.write(u_, time=t)
    end = T
    bcs = [DirichletBC(V, 0, "on_boundary")]
    params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu"}

    print('******************************** Forward solver ********************************\n')
    while t <= end:
        solve(a == 0, u,solver_parameters= params,bcs=bcs)
        u_.assign(u)
        t += timestep
        outfile.write(u_, time=t)
        print("Time:", t, "[s]")
        print(int(min(t / T * 100, 100)), "% complete")


    print('... forward problem solved...\n')

    print('...done!')
