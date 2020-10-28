import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *
from solvers.mevp_solver import *
from solvers.evp_solver import *
from solvers.forward_euler_solver import *

def evp_vp_test1(timescale=10,timestep = 10**(-1),number_of_triangles = 30,rheology="VP",advection = False,
                 solver = "FE",stabilisation = 0,subcycle = 100,output = "False"):
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
    print('\n******************************** {rheo} MODEL TEST ********************************\n'.format(rheo = rheology))

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
    #ep_dot = as_matrix([[1,0],[0,1]])

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - A))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2  * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)
    #sigma = as_matrix([[1,0],[0,1]])

    # momentum equation
    if advection == False:
        a = (inner(rho * h * (u - u_) / timestep + rho_w * C_w * sqrt(dot(u - ocean_curr,u - ocean_curr)) * (ocean_curr - u), v)) * dx
        a += inner(sigma, grad(v)) * dx


    t = 0.0
    bcs = [DirichletBC(V, 0, "on_boundary")]

    if rheology == "VP":
        forward_euler_solver(u,u_,a,bcs,t,timestep,timescale,pathname='./output/vp_evp_rheology/test1.pvd',output=output)
    elif rheology == "EVP":
        evp_solver(u, u_, a, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale,
                   pathname='./output/vp_evp_test/evp_test1.pvd', output=output)

    print('...done!')

#evp_vp_test1(timescale=10,timestep=10**(-1),rheology="EVP")

evp_vp_test1(timescale=10,timestep=10**(-1),rheology="EVP",subcycle=5)