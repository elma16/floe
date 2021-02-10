from firedrake import *
from seaice.equations import *


class SeaIceModel(object):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing):
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

        self.x, self.y = SpatialCoordinate(mesh)
        self.n = FacetNormal(mesh)
        self.V = VectorFunctionSpace(mesh, "CR", 1)
        self.U = FunctionSpace(mesh, "CR", 1)
        self.S = TensorFunctionSpace(mesh, "DG", 0)
        self.W1 = MixedFunctionSpace([self.V, self.S])
        self.W2 = MixedFunctionSpace([self.V, self.U, self.U])

    def Ice_Strength(self, h, a):
        return self.params.P_star * h * exp(-self.params.C * (1 - a))

    def strain(self, omega):
        return 0.5 * (omega + transpose(omega))

    def ep_dot(self, zeta, u):
        return 0.5 * zeta * self.strain(grad(u))

    def bcs(self, space):
        return [DirichletBC(space, values, "on_boundary") for values in self.bcs_values]

    def mom_equ(self, hh, u1, u0, p, sigma, rho, func1=None, uh=None, ocean_curr=None, rho_a=None, C_a=None, rho_w=None,
                C_w=None, geo_wind=None):
        # TODO make forcing cross product shorter
        def momentum():
            return inner(rho * hh * (u1 - u0), p) * dx

        def forcing():
            if ocean_curr is None:
                return 0
            else:
                return inner(rho * hh * cor * as_vector([u1[1] - ocean_curr[1], ocean_curr[0] - u1[0]]), p) * dx

        def ocean_stress():
            if ocean_curr is None:
                return 0
            else:
                return inner(rho_w * C_w * sqrt(dot(ocean_curr - uh, ocean_curr - uh)) * (ocean_curr - uh), p) * dx

        def air_stress():
            if geo_wind is None:
                return 0
            else:
                return inner(rho_a * C_a * sqrt(dot(geo_wind, geo_wind)) * geo_wind, p) * dx

        def alt_forcing():
            if func1 is None:
                return 0
            else:
                return inner(func1, p) * dx

        def rheo():
            return inner(sigma, grad(p)) * dx

        return momentum() - forcing() - ocean_stress() - air_stress() + alt_forcing() - rheo()

    def trans_equ(self, h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n):
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
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        v = TestFunction(self.V)

        h = Constant(1)
        a = Constant(1)

        self.zeta = 0.5 * self.Ice_Strength(h, a) / params.Delta_min
        sigma = self.ep_dot(self.zeta, self.u1)

        self.initial_conditions(self.u0, self.u1)

        sigma_exp = self.zeta * self.strain(grad(ics_values[0]))

        eqn = self.mom_equ(1, self.u1, self.u0, v, sigma, 1, div(sigma_exp))
        bcs = self.bcs(self.V)

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.initial_dump(self.u1)


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing)

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

        ep_dot = self.ep_dot(1, uh)
        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)
        zeta = 0.5 * self.Ice_Strength(h, a) / Delta

        eqn = self.mom_equ(h, u1, u0, p, sigma, params.rho, uh=uh, ocean_curr=forcing[0], rho_a=params.rho_a,
                           C_a=params.C_a, C_w=params.C_w)
        bcs = self.bcs(self.W1)

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.u1, self.s1 = self.w1.split()

        self.initial_dump(self.u1, self.s1)


class ElasticViscousPlasticStress(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing)

    u0 = Function(self.V, name="Velocity")
    u1 = Function(V, name="VelocityNext")

    sigma0 = Function(S, name="StressTensor")
    sigma1 = Function(S, name="StressTensorNext")

    uh = 0.5 * (u1 + u0)

    a = Function(U)

    v = TestFunction(V)
    w = TestFunction(S)

    u0.assign(0)

    h = Constant(1)
    a.interpolate(x / L)

    # ocean current
    ocean_curr = as_vector([0.1 * (2 * y - L) / L, -0.1 * (L - 2 * x) / L])

    # strain rate tensor
    ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))

    # ice strength
    P = P_star * h * exp(-C * (1 - a))

    Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

    zeta = 0.5 * self.Ice_Strength(h, a) / Delta
    eta = zeta * e ** (-2)

    sigma0.interpolate(2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * P * Identity(2))
    sigma1.assign(sigma0)

    def sigma_next(self, timestep, e, zeta, T, ep_dot, sigma, P):
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

    # momentum equation

    s = sigma_next(timestep, e, zeta, T, ep_dot, sigma0, P)

    sh = 0.5 * (s + sigma0)

    lm = inner(rho * h * (u1 - u0), v) * dx
    lm += timestepc * inner(rho_w * C_w * sqrt(dot(uh - ocean_curr, uh - ocean_curr)) * (uh - ocean_curr), v) * dx(
        degree=3)
    lm += timestepc * inner(sh, grad(v)) * dx

    ls = inner(w, sigma1 - s) * dx

    t = 0
    bcs = self.bcs(self.V)
    uprob = NonlinearVariationalProblem(lm, u1, bcs)
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)
    sprob = NonlinearVariationalProblem(ls, sigma1)
    ssolver = NonlinearVariationalSolver(sprob, solver_parameters=params)

    pathname = './output/implicit_evp_matrix/T={}_u_k={}.pvd'.format(timescale, timestep)

    implicit_midpoint_matrix_evp_solver(timescale, timestep, u0, t, usolver, ssolver, u1, pathname)

    print('...done!')


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params, forcing)

        self.w0 = Function(self.W2)
        self.w1 = Function(self.W2)

        u0, h0, a0 = self.w0.split()

        p, q, r = TestFunctions(self.W2)

        u0.assign(0)
        h0.assign(1)
        a0.interpolate(self.x / length)

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

        mom_eqn = self.mom_equ(hh, u1, u0, p, sigma, params.rho, uh=uh, ocean_curr=forcing[0], rho_a=params.rho_a,
                               C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, geo_wind=forcing[1])

        trans_eqn = self.trans_equ(0.5, 0.5, uh, hh, ah, h1, h0, a1, a0, q, r, self.n)
        eqn = mom_eqn + trans_eqn
        bcs = self.bcs(self.W2)

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()

        self.initial_dump(self.u1, self.h1, self.a1)
