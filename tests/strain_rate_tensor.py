import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.parameters import *
from solvers.forward_euler_solver import *

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

def strain_rate_tensor(timescale=10, timestep=10 ** (-6), number_of_triangles=35, stabilised=0,
                       transform_mesh=False, output=False):

    print('\n******************************** STRAIN RATE TENSOR ********************************\n')
    # transforming the mesh using the mapping (x,y) -> (x+y/2,y) to change the right angled triangles to equilateral triangles
    if transform_mesh:
        # want periodic bc on the sides, and dirichlet bc on the top and bottom
        mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, L, "y")
        Vc = mesh.coordinates.function_space()
        x, y = SpatialCoordinate(mesh)
        f = Function(Vc).interpolate(as_vector([x + 0.5 * y, y]))
        mesh.coordinates.assign(f)
    else:
        mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u0 = Function(V)
    u1 = Function(V)

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    h = Constant(1)

    a = Constant(1)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    # viscosities
    zeta = P / (2 * Delta_min)

    sigma = zeta / 2 * (grad(u1) + transpose(grad(u1)))

    if stabilised == 2:
        sigma = zeta / 2 * (grad(u1))

    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    # initialising at expected v value

    # u0.interpolate(v_exp)
    # u1.assign(u0)

    u0.assign(0)
    u1.assign(u0)

    sigma_exp = zeta / 2 * (grad(v_exp) + transpose(grad(v_exp)))

    R = -div(sigma_exp)

    def strain(omega):
        return 1 / 2 * (omega + transpose(omega))

    # momentum equation
    lm = inner(u1 - u0, v) * dx
    lm += timestep * inner(sigma, strain(grad(v))) * dx
    lm -= timestep * inner(R, v) * dx
    if stabilised == 1:
        lm += avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(u1), jump(v))) * dS

    t = 0.0

    if transform_mesh:
        # no compile errors, i just don't understand why mesh says 0,1 doesn't work but 1,2 does
        bcs = [DirichletBC(V, Constant(0), [1, 2])]
    else:
        bcs = [DirichletBC(V, Constant(0), "on_boundary")]

    all_u = forward_euler_solver(u1, u0, lm, bcs, t, timestep, timescale,
                                 pathname='./output/strain_rate_tensor/str_u.pvd', output=output)

    print('...done!')

    return all_u, mesh, v_exp, zeta

def toy_problem(timescale=10, timestep=10 ** (-3), stabilised=0, number_of_triangles=30, output=False, shape="Square"):
    """
    A trial toy test problem where I start off with a big square in the middle of the velocity field
    to demonstrate the nature of hyperbolic PDEs
    """
    print('\n******************************** TOY PROBLEM ********************************\n')
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u0 = Function(V)
    u1 = Function(V)

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    if shape == "Half-Plane":
        u0.interpolate(conditional(le(x, L / 2), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Square":
        u0.interpolate(conditional(le(abs(x - L / 2) + abs(y - L / 2), L / 5), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Circle":
        u0.interpolate(
            conditional(le(((x - L / 2) ** 2 + (y - L / 2) ** 2), 10000 * L), as_vector([10, 10]), as_vector([0, 0])))

    u1.assign(u0)

    h = Constant(1)

    a = Constant(1)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    # viscosities
    zeta = P / (2 * Delta_min)

    # strain rate tensor, where grad(u) is the jacobian matrix of u
    ep_dot = 1 / 2 * (grad(u1) + transpose(grad(u1)))

    eta = zeta * e ** (-2)

    # sigma = avg(CellVolume(mesh))/FacetArea(mesh)*(dot(jump(u),jump(v)))*dS
    sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)

    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))
    R = -div(sigma_exp)

    def strain(omega):
        return 0.5 * (omega + transpose(omega))

    # momentum equation
    lm = inner((u1 - u0) / timestep, v) * dx
    lm += inner(sigma, strain(grad(v))) * dx
    lm -= inner(R, v) * dx

    t = 0.0

    bcs = [DirichletBC(V, 0, "on_boundary")]

    all_u = forward_euler_solver(u1, u0, lm, bcs, t, timestep, timescale, pathname='./output/toy_test/toy.pvd',
                                 output=output)

    print('...done!')
    return all_u

