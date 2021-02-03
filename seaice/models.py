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
        self.S = TensorFunctionSpace(mesh, "DG", 0)
        self.W = MixedFunctionSpace([self.V, self.U, self.U, self.S])

        self.ocean_curr = as_vector([0.1 * (2 * self.y - length) / length, -0.1 * (length - 2 * self.x) / length])

    def strain(self, omega):
        return 0.5 * (omega + transpose(omega))

    def ep_dot(self, u, zeta):
        return zeta * SeaIceModel.strain(self, grad(u))

    def Ice_Strength(self, h, a):
        return self.params.P_star * h * exp(-self.params.C * (1 - a))

    def bcs(self, space):
        return [DirichletBC(space, values, "on_boundary") for values in self.bcs_values]

    '''
    def momentum_eqation(self, rho, hh, u1, u0, cor, p, uh, rho_a, C_a, ocean_curr, geo_wind, rho_w, C_w, sigma, Q,
                         sigma_exp):
        lm = inner(rho * hh * (u1 - u0), p) * dx
        lm -= self.timestep * inner(rho * hh * cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0] - uh[0]]), p) * dx
        lm += self.timestep * inner(rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * sqrt(
            dot(uh - ocean_curr, uh - ocean_curr)) * (ocean_curr - uh), p) * dx
        lm -= self.timestep * inner(-div(sigma_exp), p) * dx
        lm += self.timestep * inner(sigma, Q) * dx
        return lm

    # Q = SeaIceModel.strain(self, grad(p))
    # Q = grad(p)

    def transport_equations2(self, h1, h0, a1, a0, q, r):
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
        return lm
    '''

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

        zeta = 0.5 * SeaIceModel.Ice_Strength(self, h, a) / params.Delta_min

        sigma = SeaIceModel.ep_dot(self, self.u1, zeta)

        pi_x = pi / length
        v_exp = as_vector([-sin(pi_x * self.x) * sin(pi_x * self.y), -sin(pi_x * self.x) * sin(pi_x * self.y)])

        self.u0.interpolate(v_exp)
        self.u1.assign(self.u0)

        sigma_exp = SeaIceModel.ep_dot(self, v_exp, zeta)

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        lm = (inner(self.u1 - self.u0, v) + self.timestep * inner(sigma, strain(grad(v)))) * dx
        lm -= self.timestep * inner(R, v) * dx

        bcs = [DirichletBC(self.V, as_vector([0, 0]), "on_boundary")]
        uprob = NonlinearVariationalProblem(lm, bcs)
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


class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        a = Function(self.U)

        w0 = Function(self.W)

        u0, s0, h0, a0 = w0.split()

        p, q = TestFunctions(self.W)

        # initial conditions

        u0.assign(0)
        a.interpolate(self.x / length)
        h = Constant(1)

        # s0.interpolate(- 0.5 * P * Identity(2))
        # s0.assign(as_matrix([[-0.5*P,0],[0,-0.5*P]]))
        s0.assign(as_matrix([[1, 2], [3, 4]]))

        w1 = Function(self.W)
        w1.assign(w0)
        u1, s1 = split(w1)
        u0, s0 = split(w0)

        uh = 0.5 * (u0 + u1)
        sh = 0.5 * (s0 + s1)

        # strain rate tensor
        ep_dot = SeaIceModel.ep_dot(self, uh, 1)

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * SeaIceModel.Ice_Strength(self, h, a) / Delta

        uprob = NonlinearVariationalProblem(lm, SeaIceModel.momentum_equation1(u1, u0, s1, s0, p),
                                            SeaIceModel.bcs(self, self.W))
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


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params):
        super().__init__(mesh, bcs_values, ics_values, length, timestepping, params, output, solver_params)

        w0 = Function(self.W)
        w1 = Function(self.W)

        u0, h0, a0 = w0.split()

        # test functions
        p, q, r, s = TestFunctions(self.W)

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
        ep_dot = SeaIceModel.ep_dot(self, uh, 1)

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        # viscosities
        zeta = 0.5 * SeaIceModel.Ice_Strength(self, hh, ah) / Delta
        eta = zeta * params.e ** (-2)

        # internal stress tensor
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

        # initalise geo_wind
        t0 = Constant(0)

        # TODO can i turn this to a function?
        geo_wind = as_vector(
            [5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * self.x / length) * sin(
                2 * pi * self.y / length),
             5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * self.y / length) * sin(
                 2 * pi * self.x / length)])

        uprob = NonlinearVariationalProblem(SeaIceModel.mom_eqn(self, params.rho, hh, u1, u0), w1,
                                            SeaIceModel.bcs(self, self.W))
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = w1.split()

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
