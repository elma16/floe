from firedrake import *


class SeaIceModel(object):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        self.timestepping = timestepping
        self.timescale = timestepping.timescale
        self.params = params
        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output
        self.outfile = File(output.dirname)
        self.dump_count = 0
        self.dump_freq = output.dumpfreq
        self.solver_params = solver_params
        self.mesh = mesh
        self.length = length
        self.data = {'velocity': []}
        self.bcs_values = bcs_values
        self.ics_values = ics_values

        self.timestep = timestepping.timestep
        self.x, self.y = SpatialCoordinate(mesh)

        self.V = VectorFunctionSpace(mesh, "CR", 1)
        self.U = FunctionSpace(mesh, "CR", 1)

        self.ocean_curr = as_vector([0.1 * (2 * self.y - length) / length, -0.1 * (length - 2 * self.x) / length])

    def Ice_Strength(self, h, a):
        return self.params.P_star * h * exp(-self.params.C * (1 - a))


class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        v = TestFunction(self.V)

        h = Constant(1)
        a = Constant(1)

        # viscosities
        zeta = 0.5 * SeaIceModel.Ice_Strength(self, h, a) / params.Delta_min

        sigma = 0.5 * zeta * (grad(self.u1) + transpose(grad(self.u1)))

        pi_x = pi / length
        v_exp = as_vector([-sin(pi_x * self.x) * sin(pi_x * self.y), -sin(pi_x * self.x) * sin(pi_x * self.y)])

        self.u0.interpolate(v_exp)
        self.u1.assign(self.u0)

        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        # momentum equation
        lm = (inner(self.u1 - self.u0, v) + self.timestep * inner(sigma, strain(grad(v)))) * dx
        lm -= self.timestep * inner(R, v) * dx

        bcs = [DirichletBC(self.V, values, "on_boundary") for values in bcs_values]

        uprob = NonlinearVariationalProblem(lm, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.outfile.write(self.u1, time=0)

    def solve(self):
        self.usolver.solve()

    def update(self):
        self.u0.assign(self.u1)

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        S = TensorFunctionSpace(mesh, "DG", 0)

        W = MixedFunctionSpace([self.V, S])

        a = Function(self.U)

        w0 = Function(W)

        u0, s0 = w0.split()

        p, q = TestFunctions(W)

        # initial conditions

        u0.assign(0)
        a.interpolate(self.x / length)
        h = Constant(1)

        # ice strength
        P = params.P_star * h * exp(-params.C * (1 - a))

        # s0.interpolate(- 0.5 * P * Identity(2))
        # s0.assign(as_matrix([[-0.5*P,0],[0,-0.5*P]]))
        s0.assign(as_matrix([[1, 2], [3, 4]]))

        w1 = Function(W)
        w1.assign(w0)
        u1, s1 = split(w1)
        u0, s0 = split(w0)

        uh = 0.5 * (u0 + u1)
        sh = 0.5 * (s0 + s1)

        # strain rate tensor
        ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * P / Delta

        lm = (inner(p, params.rho * h * (u1 - u0)) + self.timestep * inner(grad(p), sh) + inner(q, (
                s1 - s0) + self.timestep * (
                                                                                                        0.5 * params.e ** 2 / params.T * sh + (
                                                                                                        0.25 * (
                                                                                                        1 - params.e ** 2) / params.T * tr(
                                                                                                    sh) + 0.25 * P / params.T) * Identity(
                                                                                                    2)))) * dx
        lm -= self.timestep * inner(p, params.C_w * sqrt(dot(uh - self.ocean_curr, uh - self.ocean_curr)) * (
                uh - self.ocean_curr)) * dx(
            degree=3)
        lm -= inner(q * zeta * self.timestep / params.T, ep_dot) * dx

        bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]
        uprob = NonlinearVariationalProblem(lm, w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.u1, self.s1 = w1.split()

        self.outfile.write(self.u1, self.s1, time=0)

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
            self.outfile.write(self.u1, time=t)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        W = MixedFunctionSpace([self.V, self.U, self.U])

        w0 = Function(W)
        w1 = Function(W)

        u0, h0, a0 = w0.split()

        # test functions
        p, q, r = TestFunctions(W)

        # initial conditions
        u0.assign(ics_values[0])
        h0.assign(ics_values[1])
        a0.assign(ics_values[2])
        # TODO want to assign x / l for a0 - is that possible?

        w1.assign(w0)

        u1, h1, a1 = split(w1)
        u0, h0, a0 = split(w0)

        uh = 0.5 * (u0 + u1)
        ah = 0.5 * (a0 + a1)
        hh = 0.5 * (h0 + h1)

        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        # strain rate tensor
        ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

        # ice strength
        P = params.P_star * hh * exp(-params.C * (1 - ah))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * P / Delta
        eta = zeta * params.e ** (-2)

        # internal stress tensor
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

        # initalise geo_wind
        t0 = Constant(0)

        geo_wind = as_vector(
            [5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / length) * sin(2 * pi * y / length),
             5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / length) * sin(2 * pi * x / length)])

        lm = inner(params.rho * hh * (u1 - u0), p) * dx
        lm -= self.timestep * inner(
            params.rho * hh * params.cor * as_vector([uh[1] - self.ocean_curr[1], self.ocean_curr[0]
                                                      - uh[0]]), p) * dx
        lm += self.timestep * inner(
            params.rho_a * params.C_a * dot(geo_wind, geo_wind) * geo_wind + params.rho_w * params.C_w * sqrt(
                dot(uh - self.ocean_curr, uh - self.ocean_curr)) * (
                    self.ocean_curr - uh), p) * dx
        lm += self.timestep * inner(sigma, grad(p)) * dx

        # adding the transport equations
        dh_trial = h1 - h0
        da_trial = a1 - a0

        # LHS
        lm += q * dh_trial * dx
        lm += r * da_trial * dx

        n = FacetNormal(mesh)

        un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))

        lm -= self.timestep * (hh * div(q * uh) * dx
                               - conditional(dot(uh, n) < 0, q * dot(uh, n) * h_in, 0.0) * ds
                               - conditional(dot(uh, n) > 0, q * dot(uh, n) * hh,
                                             0.0) * ds
                               - (q('+') - q('-')) * (un('+') * ah('+') - un('-') * hh('-')) * dS)

        lm -= self.timestep * (ah * div(r * uh) * dx
                               - conditional(dot(uh, n) < 0, r * dot(uh, n) * a_in, 0.0) * ds
                               - conditional(dot(uh, n) > 0, r * dot(uh, n) * ah,
                                             0.0) * ds
                               - (r('+') - r('-')) * (un('+') * ah('+') - un('-') * ah('-')) * dS)

        bcs = [DirichletBC(W.sub(bcs_values.index(values)), values, "on_boundary") for values in bcs_values]

        uprob = NonlinearVariationalProblem(lm, w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        u1, h1, a1 = w1.split()

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
