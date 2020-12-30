from firedrake import *


class SeaIceModel(object):
    """
    Defining the general class for a Sea Ice Model

    :arg timestepping:
    :arg number_of_triangles:
    :arg params:
    :output:
    :solver_params:
    """

    def __init__(self, timestepping, number_of_triangles, params, output, solver_params):
        self.timestepping = timestepping
        self.timescale = timestepping.timescale
        self.timestep = timestepping.timestep
        self.number_of_triangles = number_of_triangles
        self.params = params
        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output
        self.outfile = File(output.dirname)
        self.dump_count = 0
        self.dump_freq = output.dumpfreq
        self.solver_params = solver_params

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")

    # TODO get some shared methods into here


class StrainRateTensor(SeaIceModel):
    """
    The Strain Rate Tensor test.

    :arg timestepping:
    :arg output:
    :arg params:
    :arg stabilised:
    :arg transform_mesh:
    :arg number_of_triangles:
    """

    def __init__(self, timestepping, params, solver_params, output, stabilised='0', transform_mesh=False,
                 number_of_triangles=35):

        super().__init__(timestepping, number_of_triangles, params, output, solver_params)

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)

        self.all_u = []
        self.stabilised = stabilised
        self.transform_mesh = transform_mesh

        if transform_mesh:
            self.mesh = PeriodicSquareMesh(number_of_triangles, number_of_triangles, params.length, "y")
            Vc = self.mesh.coordinates.function_space()
            x, y = SpatialCoordinate(mesh)
            f = Function(Vc).interpolate(as_vector([x + 0.5 * y, y]))
            self.mesh.coordinates.assign(f)
        else:
            self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)

        # sea ice velocity
        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        # test functions
        self.v = TestFunction(self.V)

        x, y = SpatialCoordinate(self.mesh)

        self.h = Constant(1)
        self.a = Constant(1)

        # ice strength
        P = params.P_star * self.h * exp(-params.C * (1 - self.a))

        # viscosities
        zeta = 0.5 * P / params.Delta_min
        self.zeta = zeta

        sigma = 0.5 * zeta * (grad(self.u1) + transpose(grad(self.u1)))

        pi_x = pi / params.length
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
        self.v_exp = v_exp

        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        if transform_mesh:
            self.bcs = [DirichletBC(self.V, Constant(0), [1, 2])]
        else:
            self.bcs = [DirichletBC(self.V, Constant(0), "on_boundary")]

        # momentum equation
        self.lm = (inner(self.u1 - self.u0, self.v) + self.timestep * inner(sigma, strain(grad(self.v)))) * dx
        self.lm -= self.timestep * inner(R, self.v) * dx

        self.uprob = NonlinearVariationalProblem(self.lm, self.u1, self.bcs)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters=solver_params.srt_params)

    def solve(self, t):

        """
        Solve the equations at a given timestep
        """
        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):
        """
        Update the equations with the new values of the functions
        """

        while t < self.timescale - 0.5 * self.timestep:
            StrainRateTensor.solve(self, t)
            self.u0.assign(self.u1)
            StrainRateTensor.dump(self, t)
            t += self.timestep
            StrainRateTensor.progress(self, t)

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)

    def sp_output(self):

        t = 0

        while t < self.timescale - 0.5 * self.timestep:
            StrainRateTensor.solve(self, t)
            self.u0.assign(self.u1)
            self.all_u.append(Function(self.u1))
            t += self.timestep

        return self.all_u, self.mesh, self.v_exp, self.zeta


class Evp(SeaIceModel):
    """
    The VP/EVP test.

    :arg timestepping:
    :arg output:
    :arg params:
    :arg stabilised:
    :arg number_of_triangles:

    Solving test 2 using the implicit midpoint rule, but solving a matrix system rather than using a mixed function space.
    Solution Strategy:
    Apply the implicit midpoint rule to the coupled system of PDEs.
    Solve sigma^{n+1} in terms of sigma^{n},v^{n},v^{n+1}.
    Plug in sigma^{n+1} into the momentum equation and solve exactly for v^{n+1}.
    init = "0" for 0 initial conditions
         = "1" for manufactured solution IC.

    """

    def __init__(self, timestepping, number_of_triangles, params, output, solver_params):

        super().__init__(timestepping, number_of_triangles, params, output, solver_params)

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)
        self.S = TensorFunctionSpace(self.mesh, "DG", 0)
        self.U = FunctionSpace(self.mesh, "CR", 1)
        self.W = MixedFunctionSpace([self.V, self.S])

        self.a = Function(self.U)

        self.w0 = Function(self.W)

        self.u0, self.s0 = self.w0.split()

        x, y = SpatialCoordinate(self.mesh)

        p, q = TestFunctions(self.W)

        # initial conditions

        self.u0.assign(0)
        self.a.interpolate(x / params.length)
        self.h = Constant(1)

        # ice strength
        P = params.P_star * self.h * exp(-params.C * (1 - self.a))

        # s0.interpolate(- 0.5 * P * Identity(2))
        # s0.assign(as_matrix([[-0.5*P,0],[0,-0.5*P]]))
        # TODO fix this!
        self.s0.assign(as_matrix([[1, 2], [3, 4]]))

        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        self.u1, self.s1 = split(self.w1)
        self.u0, self.s0 = split(self.w0)

        self.uh = 0.5 * (self.u0 + self.u1)
        self.sh = 0.5 * (self.s0 + self.s1)

        # ocean current
        ocean_curr = as_vector(
            [0.1 * (2 * y - params.length) / params.length, -0.1 * (params.length - 2 * x) / params.length])

        # strain rate tensor
        ep_dot = 0.5 * (grad(self.uh) + transpose(grad(self.uh)))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * P / Delta

        self.lm = (inner(p, params.rho * self.h * (self.u1 - self.u0)) + self.timestep * inner(grad(p), self.sh) +
                   inner(q, (self.s1 - self.s0) + self.timestep * (0.5 * params.e ** 2 / params.T * self.sh +
                                                                   (0.25 * (1 - params.e ** 2) / params.T * tr(
                                                                       self.sh) + 0.25 * P / params.T) * Identity(
                               2)))) * dx
        self.lm -= self.timestep * inner(p, params.C_w * sqrt(dot(self.uh - ocean_curr, self.uh - ocean_curr)) * (
                self.uh - ocean_curr)) * dx(
            degree=3)
        self.lm -= inner(q * zeta * self.timestep / params.T, ep_dot) * dx

        self.bcs = [DirichletBC(self.W.sub(0), 0, "on_boundary")]
        self.uprob = NonlinearVariationalProblem(self.lm, self.w1, self.bcs)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.s1 = self.w1.split()

    def solve(self, t):

        """
        Solve the equations at a given timestep
        """

        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):
        """
        Update the equations with the new values of the functions
        """

        while t < self.timescale - 0.5 * self.timestep:
            Evp.solve(self, t)
            self.w0.assign(self.w1)
            t += self.timestep
            Evp.progress(self, t)

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)


class BoxTest(SeaIceModel):
    """
    The Box test.

    :arg timestepping:
    :arg output:
    :arg params:
    :arg stabilised:
    :arg transform_mesh:
    :arg number_of_triangles:
    """

    def __init__(self, timestepping, number_of_triangles, params, output, solver_params):

        super().__init__(timestepping, number_of_triangles, params, output, solver_params)

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.box_length)

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)
        self.U = FunctionSpace(self.mesh, "CR", 1)
        self.W = MixedFunctionSpace([self.V, self.U, self.U])

        self.w0 = Function(self.W)
        self.w1 = Function(self.W)

        self.u0, self.h0, self.a0 = self.w0.split()

        # test functions
        p, q, r = TestFunctions(self.W)

        x, y = SpatialCoordinate(self.mesh)

        # initial conditions
        self.u0.assign(0)
        self.h0.assign(1)
        self.a0.interpolate(x / params.box_length)

        self.w1.assign(self.w0)

        self.u1, self.h1, self.a1 = split(self.w1)
        self.u0, self.h0, self.a0 = split(self.w0)

        self.uh = 0.5 * (self.u0 + self.u1)
        self.ah = 0.5 * (self.a0 + self.a1)
        self.hh = 0.5 * (self.h0 + self.h1)

        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        # ocean current
        ocean_curr = as_vector([0.1 * (2 * y - params.box_length) / params.box_length,
                                -0.1 * (params.box_length - 2 * x) / params.box_length])

        # strain rate tensor
        ep_dot = 0.5 * (grad(self.uh) + transpose(grad(self.uh)))

        # ice strength
        P = params.P_star * self.hh * exp(-params.C * (1 - self.ah))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * P / Delta
        eta = zeta * params.e ** (-2)

        # internal stress tensor
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

        # initalise geo_wind
        self.t0 = Constant(0)

        geo_wind = as_vector(
            [5 + (sin(2 * pi * self.t0 / self.timescale) - 3) * sin(2 * pi * x / params.box_length) * sin(
                2 * pi * y / params.box_length),
             5 + (sin(2 * pi * self.t0 / self.timescale) - 3) * sin(2 * pi * y / params.box_length) * sin(
                 2 * pi * x / params.box_length)])

        self.lm = inner(params.rho * self.hh * (self.u1 - self.u0), p) * dx
        self.lm -= self.timestep * inner(
            params.rho * self.hh * params.cor * as_vector([self.uh[1] - ocean_curr[1], ocean_curr[0]
                                                           - self.uh[0]]), p) * dx
        self.lm += self.timestep * inner(
            params.rho_a * params.C_a * dot(geo_wind, geo_wind) * geo_wind + params.rho_w * params.C_w * sqrt(
                dot(self.uh - ocean_curr, self.uh - ocean_curr)) * (
                    ocean_curr - self.uh), p) * dx
        self.lm += self.timestep * inner(sigma, grad(p)) * dx

        # adding the transport equations
        dh_trial = self.h1 - self.h0
        da_trial = self.a1 - self.a0

        # LHS
        self.lm += q * dh_trial * dx
        self.lm += r * da_trial * dx

        self.n = FacetNormal(self.mesh)

        un = 0.5 * (dot(self.uh, self.n) + abs(dot(self.uh, self.n)))

        self.lm -= self.timestep * (self.hh * div(q * self.uh) * dx
                                    - conditional(dot(self.uh, self.n) < 0, q * dot(self.uh, self.n) * h_in, 0.0) * ds
                                    - conditional(dot(self.uh, self.n) > 0, q * dot(self.uh, self.n) * self.hh,
                                                  0.0) * ds
                                    - (q('+') - q('-')) * (un('+') * self.ah('+') - un('-') * self.hh('-')) * dS)

        self.lm -= self.timestep * (self.ah * div(r * self.uh) * dx
                                    - conditional(dot(self.uh, self.n) < 0, r * dot(self.uh, self.n) * a_in, 0.0) * ds
                                    - conditional(dot(self.uh, self.n) > 0, r * dot(self.uh, self.n) * self.ah,
                                                  0.0) * ds
                                    - (r('+') - r('-')) * (un('+') * self.ah('+') - un('-') * self.ah('-')) * dS)

        self.bcs = [DirichletBC(self.W.sub(0), 0, "on_boundary")]

        self.uprob = NonlinearVariationalProblem(self.lm, self.w1, self.bcs)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()

    def solve(self, t):

        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):

        while t < self.timescale - 0.5 * self.timestep:
            BoxTest.solve(self, t)
            self.w0.assign(self.w1)
            t += self.timestep
            self.t0.assign(t)
            BoxTest.progress(self, t)

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)
