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


class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        timestep = timestepping.timestep
        x, y = SpatialCoordinate(mesh)

        V = VectorFunctionSpace(mesh, "CR", 1)

        self.u0 = Function(V, name="Velocity")
        self.u1 = Function(V, name="VelocityNext")

        v = TestFunction(V)

        h = Constant(1)
        a = Constant(1)

        # ice strength
        P = params.P_star * h * exp(-params.C * (1 - a))

        # viscosities
        zeta = 0.5 * P / params.Delta_min

        sigma = 0.5 * zeta * (grad(self.u1) + transpose(grad(self.u1)))

        pi_x = pi / length
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

        self.u0.assign(ics_values[0])
        self.u1.assign(ics_values[1])
        # elif init == "1":
        #    u0.interpolate(v_exp)
        #    u1.assign(u0)

        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        # momentum equation
        lm = (inner(self.u1 - self.u0, v) + timestep * inner(sigma, strain(grad(v)))) * dx
        lm -= timestep * inner(R, v) * dx

        # bcs = [DirichletBC(V, values, "on_boundary") for values in bcs_values]
        bcs = [DirichletBC(V, as_vector([0, 0]), "on_boundary")]

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
        V = VectorFunctionSpace(mesh, "CR", 1)
        S = TensorFunctionSpace(mesh, "DG", 0)
        U = FunctionSpace(mesh, "CR", 1)
        W = MixedFunctionSpace([V, S])

        a = Function(U)

        w0 = Function(W)

        u0, s0 = w0.split()

        timestep = timestepping.timestep
        x, y = SpatialCoordinate(mesh)

        p, q = TestFunctions(W)

        # initial conditions

        u0.assign(0)
        a.interpolate(x / length)
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

        # ocean current
        ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

        # strain rate tensor
        ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * P / Delta

        lm = (inner(p, params.rho * h * (u1 - u0)) + timestep * inner(grad(p), sh) + inner(q, (s1 - s0) + timestep * (
                0.5 * params.e ** 2 / params.T * sh + (
                0.25 * (1 - params.e ** 2) / params.T * tr(sh) + 0.25 * P / params.T) * Identity(2)))) * dx
        lm -= timestep * inner(p, params.C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr)) * dx(
            degree=3)
        lm -= inner(q * zeta * timestep / params.T, ep_dot) * dx

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

        V = VectorFunctionSpace(mesh, "CR", 1)
        U = FunctionSpace(mesh, "CR", 1)
        W = MixedFunctionSpace([V, U, U])

        w0 = Function(W)
        w1 = Function(W)

        u0, h0, a0 = w0.split()

        timestep = timestepping.timestep
        x, y = SpatialCoordinate(mesh)

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

        # ocean current
        ocean_curr = as_vector([0.1 * (2 * y - params.box_length) / params.box_length,
                                -0.1 * (params.box_length - 2 * x) / params.box_length])

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
        lm -= timestep * inner(
            params.rho * hh * params.cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0]
                                                      - uh[0]]), p) * dx
        lm += timestep * inner(
            params.rho_a * params.C_a * dot(geo_wind, geo_wind) * geo_wind + params.rho_w * params.C_w * sqrt(
                dot(uh - ocean_curr, uh - ocean_curr)) * (
                    ocean_curr - uh), p) * dx
        lm += timestep * inner(sigma, grad(p)) * dx

        # adding the transport equations
        dh_trial = h1 - h0
        da_trial = a1 - a0

        # LHS
        lm += q * dh_trial * dx
        lm += r * da_trial * dx

        n = FacetNormal(mesh)

        un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))

        lm -= timestep * (hh * div(q * uh) * dx
                          - conditional(dot(uh, n) < 0, q * dot(uh, n) * h_in, 0.0) * ds
                          - conditional(dot(uh, n) > 0, q * dot(uh, n) * hh,
                                        0.0) * ds
                          - (q('+') - q('-')) * (un('+') * ah('+') - un('-') * hh('-')) * dS)

        lm -= timestep * (ah * div(r * uh) * dx
                          - conditional(dot(uh, n) < 0, r * dot(uh, n) * a_in, 0.0) * ds
                          - conditional(dot(uh, n) > 0, r * dot(uh, n) * ah,
                                        0.0) * ds
                          - (r('+') - r('-')) * (un('+') * ah('+') - un('-') * ah('-')) * dS)

        bcs = [DirichletBC(W.sub(bcs_values.index(values)), values, "on_boundary") for values in bcs_values]
        bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]

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
