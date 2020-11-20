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
                       transform_mesh=False, output=False, shape=None,init = "0"):
    """
    init = "0" for 0 initial conditions
         = "1" for manufactured solution IC.
    """
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

    timestepc = Constant(timestep)

    # optional
    if shape == "Half-Plane":
        u0.interpolate(conditional(le(x, L / 2), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Square":
        u0.interpolate(conditional(le(abs(x - L / 2) + abs(y - L / 2), L / 5), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Circle":
        u0.interpolate(
            conditional(le(((x - L / 2) ** 2 + (y - L / 2) ** 2), 10000 * L), as_vector([10, 10]), as_vector([0, 0])))

    h = Constant(1)
    a = Constant(1)

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    # viscosities
    zeta = 0.5 * P / Delta_min

    sigma = 0.5 * zeta * (grad(u1) + transpose(grad(u1)))

    if stabilised == 0:
        stab_term = 0
    elif stabilised == 1:
        stab_term = avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(u1), jump(v))) * dS
    elif stabilised == 2:
        sigma = 0.5 * zeta * grad(u1)

    pi_x = pi / L
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    if init == "0":
        # initialising at 0
        u0.assign(0)
        u1.assign(u0)
    elif init == "1":
        # initialising at expected v value (manufactured solution)
        u0.interpolate(v_exp)
        u1.assign(u0)

    sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

    R = -div(sigma_exp)

    def strain(omega):
        return 0.5 * (omega + transpose(omega))

    # momentum equation
    lm = (inner(u1 - u0, v) + timestepc * inner(sigma, strain(grad(v)))) * dx
    lm -= timestepc * inner(R, v) * dx
    lm += stab_term

    t = 0

    if transform_mesh:
        # no compile errors, i just don't understand why mesh says 0,1 doesn't work but 1,2 does
        bcs = [DirichletBC(V, Constant(0), [1, 2])]
    else:
        bcs = [DirichletBC(V, Constant(0), "on_boundary")]

    uprob = NonlinearVariationalProblem(lm, u1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    all_u, all_h, all_a = forward_euler_solver(u1, u0,usolver,t,timestep,timescale,output)

    print('...done!')

    return all_u, mesh, v_exp, zeta