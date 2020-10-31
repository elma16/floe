import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *
from solvers.mevp_solver import *
from solvers.evp_solver import *
from solvers.forward_euler_solver import *

def vp_evp_test1(timescale=10,timestep = 10**(-1),number_of_triangles = 30,rheology="VP",advection = False,
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

    solver options:
        FE = Forward Euler
        mEVP = modified EVP solver

    stabilisation:
        0 = No stabilisation
        1 = Stabilisation via algorithm
        2 = Stabilisation via change of stress tensor
    """
    print('\n******************************** {rheo} MODEL TEST ********************************\n'.format(rheo = rheology))

    L = 500000
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh,"CR",1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # mean height of sea ice
    h_ = Function(U, name="Height")
    h = Function(U, name="HeightNext")

    # sea ice concentration
    a_ = Function(U, name="Concentration")
    a = Function(U, name="ConcentrationNext")

    # test functions
    v = TestFunction(V)
    w = TestFunction(U)
    q = TestFunction(U)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u_.assign(0)

    if not advection:
        h = Constant(1)
        a.interpolate(x/L)

    if solver == "mEVP":
        beta = Constant(500)
    else:
        beta = Constant(1)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 1 / 2 * (grad(u) + transpose(grad(u)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)
    eta = zeta * e ** (-2)

    # internal stress tensor
    sigma = 2  * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)

    # momentum equation (used irrespective of advection occuring or not)

    lm = (inner(beta * rho * h * (u - u_) / timestep + rho_w * C_w * sqrt(dot(u - ocean_curr, u - ocean_curr)) * (ocean_curr - u), v)) * dx
    lm += inner(sigma, grad(v)) * dx

    if advection:
        lh = (inner((h-h_)/timestep,w))*dx
        lh -= inner(u*h,grad(w))*dx
        la = (inner((a-a_)/timestep,w))*dx
        la -= inner(u*a,grad(w))*dx

    t = 0.0
    bcs = [DirichletBC(V, 0, "on_boundary")]

    if not advection:
        if rheology == "VP":
            if solver == "FE":
                forward_euler_solver(u, u_, lm, bcs, t, timestep, timescale,
                                     pathname='./output/vp_evp_rheology/vp_test1fe.pvd', output=output)
            elif solver == "mEVP":
                mevp_solver(u, u_, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale,
                            pathname='./output/vp_evp_test/vp_test1mevp.pvd', output=output)
        elif rheology == "EVP":
            evp_solver(u, u_, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale,
                       pathname='./output/vp_evp_test/evp_test1.pvd', output=output)
    if advection:
        if rheology == "VP":
            if solver == "FE":
                forward_euler_solver(u, u_, lm, bcs, t, timestep, timescale,
                                     pathname='./output/vp_evp_rheology/vp_test1fe.pvd', output=output,
                                     advection=advection,stabilisation=stabilisation,lh=lh,la=la,h=h,h_=h_,a=a,a_=a_)
            elif solver == "mEVP":
                mevp_solver(u, u_, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale,
                            pathname='./output/vp_evp_test/vp_test1mevp.pvd', output=output)
        elif rheology == "EVP":
                evp_solver(u, u_, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale,
                           pathname='./output/vp_evp_test/evp_test1.pvd', output=output)


    print('...done!')


vp_evp_test1(timescale=10, timestep=10**(-1), rheology="VP",solver="FE",subcycle=10,output=True,advection=True)
