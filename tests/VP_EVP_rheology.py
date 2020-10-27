import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from firedrake import *
from tests.parameters import *
from solvers.mEVP_solver import EVPsolver
from solvers.solver_parameters import *

def EVP_VP_test1(T=10,timestep = 10**(-1),number_of_triangles = 30,rheology="VP",advection = "Off",solver = "FE",stabilisation = 0):
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
    print('\n******************************** VP MODEL TEST ********************************\n')

    L = 500000
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh,"CR",1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    A = Function(U)

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u_.assign(0)

    h = Constant(1)

    A.interpolate(x/L)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 1 / 2 * (grad(u) + transpose(grad(u)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - A))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)

    # momentum equation
    if advection == "Off":
        a = (inner(rho * h * (u - u_) / timestep + rho_w * C_w * sqrt(dot(u - ocean_curr,u - ocean_curr)) * (ocean_curr - u), v)) * dx
        a += inner(sigma, grad(v)) * dx

    t = 0.0
    if rheology == "VP":
        outfile = File('./output/vp_evp_test/vp_test1.pvd')
        outfile.write(u_, time=t)
        end = T
        bcs = [DirichletBC(V, 0, "on_boundary")]

        print('******************************** Forward solver ********************************\n')
        while t <= end:
            solve(a == 0, u,solver_parameters=params,bcs=bcs)
            u_.assign(u)
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / T * 100, 100)), "% complete")

        print('... forward problem solved...\n')
    if rheology == "EVP":
        outfile = File('./output/vp_evp_test/vp_test1.pvd')
        outfile.write(u_, time=t)
        end = T
        bcs = [DirichletBC(V, 0, "on_boundary")]

        print('******************************** Forward solver ********************************\n')
        while t <= end:
            solve(a == 0, u, solver_parameters=params, bcs=bcs)
            EVPsolver(sigma,ep_dot,P,T,subcycle_timestep)
            u_.assign(u)
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / T * 100, 100)), "% complete")

        print('... forward problem solved...\n')

    print('...done!')
