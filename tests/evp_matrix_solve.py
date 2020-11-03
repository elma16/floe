import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *
from solvers.solver_parameters import *

def implicit_midpoint_evp(timescale=10,timestep = 10**(-1),number_of_triangles = 30,output = "False",pathname='./output/implicit_evp/u.pvd'):
    """
    Solving test 2 using the implicit midpoint rule.

    Solution Strategy:

    Apply the implicit midpoint rule to the coupled system of PDEs.
    Solve sigma^{n+1} in terms of sigma^{n},v^{n},v^{n+1}.
    Plug in sigma^{n+1} into the momentum equation and solve exactly for v^{n+1}.
    """
    print('\n******************************** IMPLICIT EVP MODEL TEST ********************************\n')

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    S = TensorFunctionSpace(mesh, "DG", 0)
    U = FunctionSpace(mesh,"CR",1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")
    uh = 0.5*(u+u_)

    sigma_ = Function(S)
    sigma = Function(S)
    sh = (sigma_ + sigma) / 2

    a = Function(U)

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u_.assign(0)
    h = Constant(1)
    a.interpolate(x/L)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 0.5 * (grad(u) + transpose(grad(u)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 0.5 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma_ = 2  * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * P  * Identity(2)

    #using the implicit midpoint rule formulation of the tensor equation, find sigma^{n+1} in terms of sigma^{n},v^{n+1},v^{n}
    def sigma_next(timestep, e, zeta, T, ep_dot, sigma, P):
        A = (1 + (timestep * e ** 2) / (4 * T))
        B = (timestep * (1 - e ** 2)) / (8 * T)
        rhs = (1 - (timestep * e ** 2) / (4 * T)) * sigma - timestep * (timestep * (1 - e ** 2)) / (8 * T) * tr(
            sigma) * Identity(2) - timestep * P / (4 * T) * Identity(2) + zeta / T * timestep * ep_dot
        sigma00 = (((rhs[0, 0] - rhs[1, 1]) / A + (rhs[0, 0] + rhs[1, 1])) / (A + 2 * B)) / 2
        sigma11 = ((-(rhs[0, 0] - rhs[1, 1]) / A + (rhs[0, 0] + rhs[1, 1])) / (A + 2 * B)) / 2
        sigma01 = rhs[0, 1]
        sigma = as_matrix([[sigma00, sigma01], [sigma01, sigma11]])

        return sigma

    # momentum equation (used irrespective of advection occuring or not)

    lm = (inner(rho * h * (u - u_) / timestep + rho_w * C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr), v)) * dx
    lm += inner(sh, grad(v)) * dx

    t = 0.0
    bcs = [DirichletBC(V, 0, "on_boundary")]

    if output:
        outfile = File('{pathname}'.format(pathname = pathname))
        outfile.write(u_, time=t)

        print('******************************** Implicit EVP Solver ********************************\n')
        while t <= timescale:
            solve(lm == 0, u, solver_parameters=params, bcs=bcs)
            sigma = sigma_next(timestep, e, zeta, T, ep_dot, sigma_, P)
            sigma_.assign(sigma)
            u_.assign(u)
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')
    else:
        print('******************************** Implicit EVP Solver (NO OUTPUT) ********************************\n')
        while t <= timescale:
            solve(lm == 0, u, solver_parameters=params, bcs=bcs)
            sigma = sigma_next(timestep, e, zeta, T, ep_dot, sigma_, P)
            sigma_.assign(sigma)
            u_.assign(u)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')

    print('...done!')

implicit_midpoint_evp(timescale=10,timestep=1)


