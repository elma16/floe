from firedrake import *

zero_vector = Constant(as_vector([0, 0]))
zero = Constant(0)


def mom_equ(hh, u1, u0, p, sigma, rho, func1=zero_vector, uh=zero_vector, ocean_curr=zero_vector, rho_a=zero, C_a=zero,
            rho_w=zero, C_w=zero,
            geo_wind=zero_vector, cor=zero):
    def momentum():
        return inner(rho * hh * (u1 - u0), p) * dx

    def perp(u):
        return as_vector([-u[1], u[0]])

    def forcing():
        return inner(rho * hh * cor * perp(ocean_curr - u1), p) * dx

    def stress(density, drag, func):
        return inner(density * drag * sqrt(dot(func, func)) * func, p) * dx(degree=3)

    def alt_forcing():
        return inner(func1, p) * dx

    def rheo():
        return inner(sigma, grad(p)) * dx

    return momentum() - forcing() - stress(rho_w, C_w, ocean_curr - uh) - stress(rho_a, C_a,
                                                                                 geo_wind) + alt_forcing() - rheo()


def stab(mesh, v, test):
    return avg(CellVolume(mesh)) / FacetArea(mesh) * (dot(jump(v), jump(test))) * dS


class SeaIceModel(object):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                 stabilised):
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
        self.forcing = forcing
        self.ics_values = ics_values
        self.stabilised = stabilised

        self.x, self.y = SpatialCoordinate(mesh)
        self.n = FacetNormal(mesh)
        self.V = VectorFunctionSpace(mesh, "CR", 1)
        self.U = FunctionSpace(mesh, "CR", 1)
        self.S = TensorFunctionSpace(mesh, "DG", 0)
        self.W1 = MixedFunctionSpace([self.V, self.S])
        self.W2 = MixedFunctionSpace([self.V, self.U, self.U])

    def Ice_Strength(self, h, a):
        return self.params.P_star * h * exp(-self.params.C * (1 - a))

    def zeta(self, h, a, delta):
        return 0.5 * self.Ice_Strength(h, a) / delta

    def strain(self, omega):
        return 0.5 * (omega + transpose(omega))

    def delta(self, u):
        return sqrt(self.params.Delta_min ** 2 + 2 * self.params.e ** (-2) * inner(dev(self.strain(grad(u))),
                                                                                   dev(self.strain(grad(u)))) + tr(
            self.strain(grad(u))) ** 2)

    def bcs(self, space):
        return [DirichletBC(space, values, "on_boundary") for values in self.bcs_values]

    def solve(self, *args):
        for solvers in args:
            solvers.solve()

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
        for variables in args:
            ix = args.index(variables)
            if type(self.ics_values[ix // 2]) == int:
                variables.assign(self.ics_values[ix // 2])
            else:
                variables.interpolate(self.ics_values[ix // 2])

    def initial_dump(self, *args):
        return self.outfile.write(*args, time=0)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")


class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                 stabilised, simple):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                         stabilised)

        self.simple = simple

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        v = TestFunction(self.V)

        h = Constant(1)
        a = Constant(1)

        ep_dot = self.strain(grad(self.u1))

        if simple:
            zeta = self.zeta(h, a, params.Delta_min)
            sigma = zeta * ep_dot
        else:
            zeta = self.zeta(h, a, self.delta(self.u1))
            eta = zeta * params.e ** (-2)
            sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * self.Ice_Strength(h,
                                                                                                         a) * Identity(
                2)

        self.initial_conditions(self.u0, self.u1)

        sigma_exp = zeta * self.strain(grad(ics_values[0]))

        eqn = mom_equ(1, self.u1, self.u0, v, sigma, 1, div(sigma_exp))
        if self.stabilised:
            eqn += stab(mesh, self.u1, v)
        bcs = self.bcs(self.V)

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.initial_dump(self.u1)


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                 stabilised):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                         stabilised)

        self.w0 = Function(self.W1)
        self.w1 = Function(self.W1)
        a = Function(self.U)
        h = Constant(1)

        u0, s0 = self.w0.split()
        p, q = TestFunctions(self.W1)

        u0.assign(ics_values[0])
        a.interpolate(ics_values[1])
        s0.assign(ics_values[2])

        self.w1.assign(self.w0)
        u1, s1 = split(self.w1)
        u0, s0 = split(self.w0)

        uh = 0.5 * (u0 + u1)
        sh = 0.5 * (s0 + s1)

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(h, a, self.delta(uh))

        # TODO clean up this equation
        eqn = mom_equ(h, u1, u0, p, sh, params.rho, uh=uh, ocean_curr=forcing[0], rho_w=params.rho_w,
                      rho_a=params.rho_a, C_a=params.C_a, C_w=params.C_w)
        eqn += self.timestep * inner(q, (s1 - s0) + self.timestep * (0.5 * params.e ** 2 / params.T * sh + (
                0.25 * (1 - params.e ** 2) / params.T * tr(sh) + 0.25 * self.Ice_Strength(h,
                                                                                          a) / params.T) * Identity(
            2))) * dx
        eqn -= inner(q * zeta * self.timestep / params.T, ep_dot) * dx
        if self.stabilised is True:
            eqn += stab(mesh, uh, p)
        bcs = self.bcs(self.W1.sub(0))

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.u1, self.s1 = self.w1.split()

        self.initial_dump(self.u1)


class ElasticViscousPlasticStress(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                 stabilised):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                         stabilised)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        self.sigma0 = Function(self.S, name="StressTensor")
        self.sigma1 = Function(self.S, name="StressTensorNext")

        uh = 0.5 * (self.u1 + self.u0)

        a = Function(self.U)

        v = TestFunction(self.V)
        w = TestFunction(self.S)

        self.u0.assign(ics_values[0])
        h = Constant(1)
        a.interpolate(ics_values[1])

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(h, a, self.delta(uh))
        eta = zeta * params.e ** (-2)

        self.sigma0.interpolate(
            2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * self.Ice_Strength(h, a) * Identity(2))
        self.sigma1.assign(self.sigma0)

        def sigma_next(timestep, e, zeta, T, ep_dot, sigma, P):
            A = 1 + 0.25 * (timestep * e ** 2) / T
            B = timestep * 0.125 * (1 - e ** 2) / T
            rhs = (1 - (timestep * e ** 2) / (4 * T)) * sigma - timestep / T * (
                    0.125 * (1 - e ** 2) * tr(sigma) * Identity(2) - 0.25 * P * Identity(2) + zeta * ep_dot)
            C = (rhs[0, 0] - rhs[1, 1]) / A
            D = (rhs[0, 0] + rhs[1, 1]) / (A + 2 * B)
            sigma00 = 0.5 * (C + D)
            sigma11 = 0.5 * (D - C)
            sigma01 = rhs[0, 1]
            sigma = as_matrix([[sigma00, sigma01], [sigma01, sigma11]])

            return sigma

        s = sigma_next(self.timestep, params.e, zeta, params.T, ep_dot, self.sigma0, self.Ice_Strength(h, a))

        sh = 0.5 * (s + self.sigma0)

        equ = mom_equ(h, self.u1, self.u0, v, sh, params.rho, uh=uh, ocean_curr=forcing[0], rho_a=params.rho_a,
                      C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, cor=params.cor)
        tensor_eqn = inner(self.sigma1 - s, w) * dx
        bcs = self.bcs(self.V)

        uprob = NonlinearVariationalProblem(equ, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)
        sprob = NonlinearVariationalProblem(tensor_eqn, self.sigma1)
        self.ssolver = NonlinearVariationalSolver(sprob, solver_parameters=solver_params.srt_params)


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                 stabilised):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing,
                         stabilised)

        self.w0 = Function(self.W2)
        self.w1 = Function(self.W2)

        u0, h0, a0 = self.w0.split()

        p, q, r = TestFunctions(self.W2)

        u0.assign(ics_values[0])
        h0.assign(ics_values[1])
        a0.interpolate(ics_values[2])

        self.w1.assign(self.w0)

        u1, h1, a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        uh = 0.5 * (u0 + u1)
        ah = 0.5 * (a0 + a1)
        hh = 0.5 * (h0 + h1)

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(hh, ah, self.delta(uh))
        eta = zeta * params.e ** (-2)
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - self.Ice_Strength(hh, ah) * 0.5 * Identity(
            2)

        def trans_equ(h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n):
            def in_term(var1, var2, test):
                trial = var2 - var1
                return test * trial * dx

            def upwind_term(var1, bc_in, test):
                un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
                return self.timestep * (var1 * div(test * uh) * dx
                                        - conditional(dot(uh, n) < 0, test * dot(uh, n) * bc_in, 0.0) * ds
                                        - conditional(dot(uh, n) > 0, test * dot(uh, n) * var1, 0.0) * ds
                                        - (test('+') - test('-')) * (un('+') * ah('+') - un('-') * var1('-')) * dS)

            return in_term(h0, h1, q) + in_term(a0, a1, r) + upwind_term(hh, h_in, q) + upwind_term(ah, a_in, r)

        eqn = mom_equ(hh, u1, u0, p, sigma, params.rho, uh=uh, ocean_curr=forcing[0], rho_a=params.rho_a,
                      C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, geo_wind=forcing[1], cor=params.cor)
        trans_eqn = trans_equ(0.5, 0.5, uh, hh, ah, h1, h0, a1, a0, q, r, self.n)
        eqn += trans_eqn
        bcs = self.bcs(self.W2.sub(0))

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()

        self.initial_dump(self.u1, self.h1, self.a1)
