from firedrake import *


class SeaIceModel(object):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        self.timestepping = timestepping
        self.timestep = timestepping.timestep
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
        self.bcs_values = bcs_values
        self.ics_values = ics_values

        self.x, self.y = SpatialCoordinate(mesh)
        self.V = VectorFunctionSpace(mesh, "CR", 1)
        self.U = FunctionSpace(mesh, "CR", 1)
        self.S = TensorFunctionSpace(mesh, "DG", 0)
        self.W = MixedFunctionSpace([self.V, self.U, self.U])

    def Ice_Strength(self, h, a):
        return self.params.P_star * h * exp(-self.params.C * (1 - a))

    def strain(self, omega):
        return 0.5 * (omega + transpose(omega))

    def ep_dot(self, zeta, u):
        return 0.5 * zeta * self.strain(grad(u))

    def bcs(self, space):
        return [DirichletBC(space, values, "on_boundary") for values in self.bcs_values]

    def mom_equ(self, rho, h, u1, u0, v, sigma, sigma_exp, func_v):
        def mass():
            return inner(rho * h * (u1 - u0), v) * dx

        def forcing():
            return self.timestep * inner(-div(sigma_exp), v) * dx

        def forcing2(geo_wind, ocean_curr, uh, v, rho_a, C_a, rho_w, C_w):
            return self.timestep * inner(rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * sqrt(
                dot(uh - ocean_curr, uh - ocean_curr)) * (ocean_curr - uh), v) * dx

        def forcing3(rho, hh, cor, uh, ocean_curr):
            return self.timestep * inner(rho * hh * cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0] - uh[0]]),
                                         v) * dx

        def rheo():
            return self.timestep * inner(sigma, func_v) * dx

        return mass() - forcing() + rheo()

    def mom_eqn3(self):

        return lm

    def solve(self, usolver):
        usolver.solve()

    def update(self, old_var, new_var):
        old_var.assign(new_var)

    def dump(self, new_var, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(new_var, time=t)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")


class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        v = TestFunction(self.V)

        h = Constant(1)
        a = Constant(1)

        self.zeta = 0.5 * self.Ice_Strength(h, a) / params.Delta_min
        sigma = self.ep_dot(self.zeta, self.u1)

        self.u0.interpolate(ics_values[0])
        self.u1.assign(self.u0)

        sigma_exp = self.zeta * self.strain(grad(ics_values[0]))

        eqn = self.mom_equ(1, 1, self.u1, self.u0, v, sigma, sigma_exp, self.strain(grad(v)))
        bcs = self.bcs(self.V)

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.outfile.write(self.u1, time=0)


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        a = Function(self.U)
        w0 = Function(self.W)
        u0, s0 = w0.split()
        p, q = TestFunctions(self.W)

        # initial conditions
        u0.assign(0)
        a.interpolate(self.x / length)
        h = Constant(1)
        s0.assign(as_matrix([[1, 2], [3, 4]]))

        w1 = Function(self.W)
        w1.assign(w0)
        u1, s1 = split(w1)
        u0, s0 = split(w0)

        uh = 0.5 * (u0 + u1)
        sh = 0.5 * (s0 + s1)

        ep_dot = self.ep_dot(1, uh)
        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)
        zeta = 0.5 * self.Ice_Strength(h, a) / Delta

        eqn = self.mom_equ()
        bcs = self.bcs(self.W)

        uprob = NonlinearVariationalProblem(eqn, w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.u1, self.s1 = w1.split()

        self.outfile.write(self.u1, self.s1, time=0)


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        self.w0 = Function(self.W)
        self.w1 = Function(self.W)

        u0, h0, a0 = self.w0.split()

        # test functions
        p, q, r = TestFunctions(self.W)

        # initial conditions
        u0.assign(ics_values[0])
        h0.assign(ics_values[1])
        a0.assign(ics_values[2])
        # TODO want to assign x / l for a0 - is that possible?

        self.w1.assign(self.w0)

        u1, h1, a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        uh = 0.5 * (u0 + u1)
        ah = 0.5 * (a0 + a1)
        hh = 0.5 * (h0 + h1)

        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        ep_dot = self.ep_dot(1, uh)
        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)
        zeta = 0.5 * self.Ice_Strength(hh, ah) / Delta
        eta = zeta * params.e ** (-2)

        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - self.Ice_Strength(hh, ah) * 0.5 * Identity(
            2)

        # initalise geo_wind
        t0 = Constant(0)

        geo_wind = as_vector(
            [5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * self.x / length) * sin(
                2 * pi * self.y / length),
             5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * self.y / length) * sin(
                 2 * pi * self.x / length)])

        def transport_equation():
            dh_trial = h1 - h0
            da_trial = a1 - a0
            lm = 0

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

        eqn = self.mom_equ()
        bcs = self.bcs(self.W)

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        u1, h1, a1 = self.w1.split()

        self.outfile.write(self.u1, self.h1, self.a1, time=0)
