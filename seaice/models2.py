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
        self.all_u = []
        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)
        self.x, self.y = SpatialCoordinate(self.mesh)
        self.data = {'velocity': []}

        # defining the function spaces
        self.V = VectorFunctionSpace(self.mesh, "CR", 1)
        self.U = FunctionSpace(self.mesh, "CR", 1)
        self.W = MixedFunctionSpace([self.V, self.U, self.U])

        self.w0 = Function(self.W)
        self.w1 = Function(self.W)

        self.u0, self.h0, self.a0 = self.w0.split()

        # test functions
        p, q, r = TestFunctions(self.W)

        # TODO initial conditions
        self.u0.assign(0)
        self.h0.assign(1)
        self.a0.interpolate(self.x / params.box_length)

        self.w1.assign(self.w0)

        self.u1, self.h1, self.a1 = split(self.w1)
        self.u0, self.h0, self.a0 = split(self.w0)

        self.uh = 0.5 * (self.u0 + self.u1)
        self.ah = 0.5 * (self.a0 + self.a1)
        self.hh = 0.5 * (self.h0 + self.h1)

        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        # TODO FORCING ocean current
        ocean_curr = as_vector([0.1 * (2 * self.y - params.box_length) / params.box_length,
                                -0.1 * (params.box_length - 2 * self.x) / params.box_length])

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

        def geo_wind(x, y):
            return as_vector(
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

        self.outfile.write(self.u1, self.h1, self.a1, time=0)

    def solve(self):
        self.usolver.solve()

    def update(self):
        self.w0.assign(self.w1)

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, self.h1, self.a1, time=t)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")
