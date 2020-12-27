import os, sys, inspect
from firedrake import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config.config import *


def srt(timescale=10, timestep=10 ** (-6), number_of_triangles=35, stabilised=0,
        transform_mesh=False, output=False, shape=None, init="0"):
    """
    init = "0" for 0 initial conditions
         = "1" for manufactured solution IC.
    """
    print('\n******************************** STRAIN RATE TENSOR ********************************\n')
    # transforming the mesh using the mapping (x,y) -> (x+y/2,y) to change the right angled triangles to equilateral
    # triangles
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
    u0 = Function(V, name="Velocity")
    u1 = Function(V, name="VelocityNext")

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
        u0.assign(0)
        u1.assign(u0)
    elif init == "1":
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
        bcs = [DirichletBC(V, Constant(0), [1, 2])]
    else:
        bcs = [DirichletBC(V, Constant(0), "on_boundary")]

    uprob = NonlinearVariationalProblem(lm, u1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    all_u, all_h, all_a = forward_euler_solver(u1, u0, usolver, t, timestep, timescale, output)

    print('...done!')

    return all_u, mesh, v_exp, zeta


# TODO define class SeaIceModel(object):

class StrainRateTensor(object):
    def __init__(self, timestepping, number_of_triangles, stabilised, transform_mesh, output, shape,
                 params):

        """
        Given the initial conditions, create the equations with the variables given

        TODO add stabilised, transform mesh
        """
        self.timestepping = timestepping
        self.number_of_triangles = number_of_triangles
        self.stabilised = stabilised
        self.params = params

        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)

        # sea ice velocity
        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        # test functions
        self.v = TestFunction(self.V)

        x, y = SpatialCoordinate(self.mesh)

        if shape == "Half-Plane":
            self.u0.interpolate(conditional(le(x, params.length / 2), as_vector([10, 10]), as_vector([0, 0])))
        elif shape == "Square":
            self.u0.interpolate(
                conditional(le(abs(x - params.length / 2) + abs(y - params.length / 2), params.length / 5),
                            as_vector([10, 10]), as_vector([0, 0])))
        elif shape == "Circle":
            self.u0.interpolate(
                conditional(le(((x - params.length / 2) ** 2 + (y - params.length / 2) ** 2), 10000 * params.length),
                            as_vector([10, 10]),
                            as_vector([0, 0])))

        self.h = Constant(1)
        self.a = Constant(1)

        # ice strength
        P = params.P_star * self.h * exp(-params.C * (1 - self.a))

        # viscosities
        zeta = 0.5 * P / params.Delta_min

        sigma = 0.5 * zeta * (grad(self.u1) + transpose(grad(self.u1)))

        pi_x = pi / params.length
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        self.bcs = [DirichletBC(self.V, Constant(0), "on_boundary")]

        # momentum equation
        lm = (inner(self.u1 - self.u0, self.v) + timestepping.timestep * inner(sigma, strain(grad(self.v)))) * dx
        lm -= timestepping.timestep * inner(R, self.v) * dx

    def solve(self):

        """
        Solve the equations at a given timestep
        """
        t = 0
        uprob = NonlinearVariationalProblem(lm, self.u1, self.bcs)
        usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)
        all_u, all_h, all_a = forward_euler_solver(self.u1, self.u0, usolver, t, timestep, timescale, output)
        return 0

    def update(self):
        """
        Update the equations with the new values of the functions
        """
        return 0

    def dump(self, t):
        """
        Output the diagnostics
        """
        all_u = []
        all_h = []
        all_a = []
        ndump = 10
        dumpn = 0

        pathname = "./output/strain_rate_tensor/test_T={}_k={}_N={}_stab={}.pvd"

        outfile = File(pathname)

        print('******************************** Forward solver ********************************\n')
        if advection:
            outfile.write(u1, h1, a1, time=t)
            while t < timescale - 0.5 * timestep:
                usolver.solve()
                u0.assign(u1)
                hsolver.solve()
                h0.assign(h1)
                asolver.solve()
                a0.assign(a1)
                all_u.append(Function(u1))
                all_h.append(Function(h1))
                all_a.append(Function(a1))
                t += timestep
                dumpn += 1
                if dumpn == ndump:
                    dumpn -= ndump
                    outfile.write(u1, h1, a1, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
        else:
            outfile.write(u1, time=t)
            while t < timescale - 0.5 * timestep:
                usolver.solve()
                u0.assign(u1)
                all_u.append(Function(u1))
                t += timestep
                dumpn += 1
                if dumpn == ndump:
                    dumpn -= ndump
                    outfile.write(u1, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

        print('... forward problem solved...\n')
        return all_u, all_h, all_a
        return 0
