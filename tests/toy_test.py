import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *
from solvers.forward_euler_solver import *

def toy_problem(timescale=10,timestep=10**(-3),stabilised=0,number_of_triangles=30,output=False,shape = "Square"):
    """
    A trial toy test problem where I start off with a big square in the middle of the velocity field
    to demonstrate the nature of hyperbolic PDEs
    """
    print('\n******************************** TOY PROBLEM ********************************\n')
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    if shape == "Half-Plane":
        u_.interpolate(conditional(le(x,L/2),as_vector([10,10]),as_vector([0,0])))
    elif shape == "Square":
        u_.interpolate(conditional(le(abs(x-L/2)+abs(y-L/2),L/5), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Circle":
        u_.interpolate(conditional(le(((x-L/2)**2+(y-L/2)**2),10000*L), as_vector([10, 10]), as_vector([0, 0])))

    u.assign(u_)

    h = Constant(1)

    a = Constant(1)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    # viscosities
    zeta = P / (2 * Delta_min)

    sigma = avg(CellVolume(mesh))/FacetArea(mesh)*(dot(jump(u),jump(v)))*dS

    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    sigma_exp = zeta / 2 * (grad(v_exp) + transpose(grad(v_exp)))
    R = -div(sigma_exp)

    def strain(omega):
        return 1 / 2 * (omega + transpose(omega))

    # momentum equation
    lm = (inner((u - u_) / timestep, v) + inner(sigma, strain(grad(v)))) * dx
    lm -= inner(R, v) * dx

    t = 0.0

    bcs = [DirichletBC(V, 0, "on_boundary")]


    all_u = forward_euler_solver(u, u_, lm, bcs, t, timestep, timescale,pathname='./output/toy_test/toy.pvd', output=output)

    print('...done!')
    return all_u

#toy_problem(timescale=1,timestep=10**(-1),output=True)