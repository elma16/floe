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

    def mom_equ(self, uh, hh, u1, u0, v, sigma, sigma_exp, rho, func_v, ocean_curr, geo_wind):
        def mass():
            return inner(rho * hh * (u1 - u0), v) * dx

        def forcing():
            return self.timestep * inner(-div(sigma_exp), v) * dx

        def forcing2(rho_a, C_a, rho_w, C_w):
            return self.timestep * inner(rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * sqrt(
                dot(uh - ocean_curr, uh - ocean_curr)) * (ocean_curr - uh), v) * dx

        def forcing3(cor):
            return self.timestep * inner(rho * hh * cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0] - uh[0]]),
                                         v) * dx

        def rheo():
            return self.timestep * inner(sigma, func_v) * dx

        return mass() - forcing() + rheo()

    def trans_equ(self, h_in, a_in, uh, hh, h1, a1, h0, a0, q, r, ah):
        dh_trial = h1 - h0
        da_trial = a1 - a0
        n = FacetNormal(self.mesh)
        un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))

        def in_term(test, trial):
            return test * trial * dx

        def upwind_term(var1, var2, var3, var4, bc_in, test, normal):
            return self.timestep * (var1 * div(test * var3) * dx
                                    - conditional(dot(var3, normal) < 0, test * dot(var3, normal) * bc_in, 0.0) * ds
                                    - conditional(dot(var3, normal) > 0, test * dot(var3, normal) * var1, 0.0) * ds
                                    - (test('+') - test('-')) * (var4('+') * var2('+') - var4('-') * var1('-')) * dS)

        return in_term(q, dh_trial) + in_term(r, da_trial) + upwind_term(hh, ah, uh, un, h_in, q, n) + upwind_term(ah, ah, uh, un, a_in, r, n)

    def solve(self, usolver):
        usolver.solve()

    def update(self, old_var, new_var):
        old_var.assign(new_var)

    def dump(self, var1, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(var1, time=t)

    def initial_conditions(self, *args):
        for vars in args:
            ix = args.index(vars)
            if type(self.ics_values[ix//2]) == int:
                vars.assign(self.ics_values[ix//2])
            else:
                vars.interpolate(self.ics_values[ix//2])

    def formulate_problem(self, eqn, var, bcs, solver_params):
        return

    def inital_dump(self, *args):
        return self.outfile.write(*args, time=0)

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

        self.initial_conditions(self.u0, self.u1)

        sigma_exp = self.zeta * self.strain(grad(ics_values[0]))

        eqn = self.mom_equ(1, 1, self.u1, self.u0, v, sigma, sigma_exp, 1, self.strain(grad(v)), 1, 1)
        bcs = self.bcs(self.V)

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.inital_dump(self.u1)


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        a = Function(self.U)
        w0 = Function(self.W)
        u0, s0 = w0.split()
        p, q = TestFunctions(self.W)

        u0.assign(ics_values[0])
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

        self.inital_dump(self.u1, self.s1)


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

        ep_dot = self.ep_dot(1, uh)
        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)
        zeta = 0.5 * self.Ice_Strength(hh, ah) / Delta
        eta = zeta * params.e ** (-2)

        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - self.Ice_Strength(hh, ah) * 0.5 * Identity(
            2)

        mom_eqn = self.mom_equ()
        tran_eqn = self.trans_equ(0.5, 0.5)
        eqn = mom_eqn + tran_eqn
        bcs = self.bcs(self.W)

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()

        self.inital_dump(self.u1, self.h1, self.a1)
