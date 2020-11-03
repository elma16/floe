import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *

def implicit_midpoint(number_of_triangles=35,timestep=10,timescale=100):
    """
    Solving the EVP model using an implicit midpoint method.
    """
    print('\n******************************** IMPLICIT EVP MODEL TEST ********************************\n')

    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    S = TensorFunctionSpace(mesh, "DG", 0)
    U = FunctionSpace(mesh,"CR",1)
    W = MixedFunctionSpace([V,S])

    a = Function(U)

    w0 = Function(W)

    u0,s0 = w0.split()

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    u0.assign(0)
    a.interpolate(x / L)

    # now we solve for s0, given u0

    s0.assign(as_matrix([[1,2],[3,4]]))

    # now we solve the whole system

    p, q = TestFunctions(W)

    w1 = Function(W)
    w1.assign(w0)
    u1, s1 = split(w1)
    u0, s0 = split(w0)

    h = Constant(1)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 1 / 2 * (grad(u0) + transpose(grad(u0)))

    # deviatoric part of the strain rate tensor
    ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

    # viscosities
    zeta = P / (2 * Delta)

    uh = 0.5*(u1+u0)
    sh = 0.5*(s1+s0)
    ep_doth = 1 / 2 * (grad(uh) + transpose(grad(uh)))

    #constructing the equations used

    lm = inner(p,rho*h*(u1-u0))*dx
    lm += (timestep*inner(grad(p),sh))*dx
    lm -= (timestep*inner(p,C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr)))*dx
    lm += inner(q,s1-s0+timestep*(e**2/(2*T)*sh+((1-e**2)/(4*T)*tr(sh)+P/(4*T))*Identity(2)))*dx
    lm -= inner(q*zeta*timestep/T,ep_doth)*dx

    bcs = [DirichletBC(V, 0, "on_boundary")]

    uprob = NonlinearVariationalProblem(lm,w1,bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters= {'mat_type': 'aij','ksp_type': 'preonly','pc_type': 'lu'})

    u0, s0 = w0.split()
    u1, s1 = w1.split()

    ufile = File('./output/evp_alt/u.pvd')
    t = 0.0
    ufile.write(u1, time=t)
    all_us = []

    ndump = 10
    dumpn = 0

    while t < timescale - 0.5 * timestep:
        t += timestep

        usolver.solve()
        w0.assign(w1)

        dumpn += 1
        if dumpn == ndump:
            dumpn -= ndump
            ufile.write(u1, time=t)
            all_us.append(Function(u1))


    print('...done!')

implicit_midpoint()

