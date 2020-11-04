import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.parameters import *
from solvers.forward_euler_solver import *

def strain_rate_tensor(timescale=10,timestep=10**(-6),number_of_triangles=35,stabilised=0,
                       transform_mesh = False,output=False):
    """
    from Mehlmann and Korn, 2020
    Section 4.2
    L = 500000
    pi_x = pi_y = pi/L
    By construction, the analytical solution is
        v_1 = -sin(pi_x*x)*sin(pi_y*y)
        v_2 = -sin(pi_x*x)*sin(pi_y*y)
    zeta = P/2*Delta_min

    number_of_triangles: paper's value for 3833 edges is between 35,36.

    stabilised = {0,1,2}
    0 - unstabilised (default option)
    1 - stabilised (change the form of the stress tensor)
    2 - stabilised (via the a velocity jump algorithm)
    """
    print('\n******************************** STRAIN RATE TENSOR ********************************\n')
    # transforming the mesh using the mapping (x,y) -> (x+y/2,y) to change the right angled triangles to equilateral triangles
    if transform_mesh:
        # want periodic bc on the sides, and dirichlet bc on the top and bottom
        mesh = PeriodicSquareMesh(number_of_triangles,number_of_triangles,L,"y")
        Vc = mesh.coordinates.function_space()
        x, y = SpatialCoordinate(mesh)
        f = Function(Vc).interpolate(as_vector([x + 0.5 * y, y]))
        mesh.coordinates.assign(f)
    else:
        mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    h = Constant(1)

    a = Constant(1)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    # viscosities
    zeta = P / (2 * Delta_min)

    sigma = zeta / 2 * (grad(u) + transpose(grad(u)))

    if stabilised == 2:
        sigma = zeta / 2 * (grad(u))

    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    #initialising at expected v value

    #u_.interpolate(v_exp)
    #u.assign(u_)

    u_.assign(0)
    u.assign(u_)

    sigma_exp = zeta / 2 * (grad(v_exp) + transpose(grad(v_exp)))

    R = -div(sigma_exp)

    def strain(omega):
        return 1 / 2 * (omega + transpose(omega))

    # momentum equation
    lm = (inner((u - u_) / timestep, v) + inner(sigma, strain(grad(v)))) * dx
    lm -= inner(R, v) * dx
    if stabilised == 1:
        lm += avg(CellVolume(mesh))/FacetArea(mesh)*(dot(jump(u),jump(v)))*dS


    t = 0.0

    if transform_mesh:
        # no compile errors, i just don't understand why mesh says 0,1 doesn't work but 1,2 does
        bcs = [DirichletBC(V, Constant(0), [1,2])]
    else:
        bcs = [DirichletBC(V, Constant(0), "on_boundary")]

    all_u = forward_euler_solver(u, u_, lm, bcs, t, timestep, timescale,pathname='./output/strain_rate_tensor/str_u.pvd', output=output)

    print('...done!')

    return all_u,mesh,v_exp,zeta

strain_rate_tensor(10,1,stabilised=0,output=True)