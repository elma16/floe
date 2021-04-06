from firedrake import *

zero_vector = Constant(as_vector([0, 0]))
zero = Constant(0)

# TODO consider further refactoring to reduce the number of arguments passed into each of these functions and classes.
# could merge params and solver params?

def mom_equ(hh, u1, u0, p, sigma, rho, uh=zero_vector, ocean_curr=zero_vector, rho_a=zero, C_a=zero, rho_w=zero,
            C_w=zero,
            geo_wind=zero_vector, cor=zero,ind=1):
    def momentum_term():
        return inner(rho * hh * (u1 - u0), p) * dx

    def perp(u):
        return as_vector([-u[1], u[0]])

    def forcing_term():
        return inner(rho * hh * cor * perp(ocean_curr - uh), p) * dx

    def stress_term(density, drag, func):
        return inner(density * drag * sqrt(dot(func, func)) * func, p) * dx(degree=3)

    def rheology_term():
        return inner(sigma, grad(p)) * dx

    return ind * momentum_term() - forcing_term() - stress_term(rho_w, C_w, ocean_curr - uh) - stress_term(rho_a, C_a,
                                                                                                     geo_wind) + rheology_term()


def stabilisation_term(alpha, zeta, mesh, v, test):
    e = avg(CellVolume(mesh)) / FacetArea(mesh)
    return 2 * alpha * zeta / e * (dot(jump(v), jump(test))) * dS


class SeaIceModel(object):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):

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
        self.conditions = conditions
        
        family = conditions['family'] 
        self.x, self.y = SpatialCoordinate(mesh)
        self.n = FacetNormal(mesh)
        self.V = VectorFunctionSpace(mesh, family, 1)
        self.U = FunctionSpace(mesh, family, 1)
        self.U1 = FunctionSpace(mesh, 'DG' ,1)
        self.S = TensorFunctionSpace(mesh, 'DG', 0)
        self.W1 = MixedFunctionSpace([self.V, self.S])
        self.W2 = MixedFunctionSpace([self.V, self.U1, self.U1])
        self.W3 = MixedFunctionSpace([self.V, self.S, self.U1, self.U1])

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
        return [DirichletBC(space, values, "on_boundary") for values in self.conditions['bc']]

    def solve(self, *args):
        for solvers in args:
            solvers.solve()

    def update(self, old_var, new_var):
        old_var.assign(new_var)

    def dump(self, *args, t):
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(*args, time=t)
            
# TODO fix this if possible to iterate over the dictionary
    def initial_conditions(self, *args):
        for variables in args:
            ix = args.index(variables)
            if type(self.conditions['ic'][ix // 2]) == int:
                variables.assign(self.conditions['ic'][ix // 2])
            else:
                variables.interpolate(self.conditions['ic'][ix // 2])

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")


class SimpleViscousPlastic(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        v = TestFunction(self.V)

        h = Constant(1)
        a = Constant(1)

        ep_dot = self.strain(grad(self.u1))

        zeta = self.zeta(h, a, params.Delta_min)
        sigma = zeta * ep_dot
        # TODO want to move this to example/
        sigma_exp = zeta * self.strain(grad(conditions['ic']['u']))
        eqn = mom_equ(h, self.u1, self.u0, v, sigma, 1)
        eqn -= inner(div(sigma_exp), v) * dx

        self.u0.interpolate(conditions['ic']['u'])
        self.u1.assign(self.u0)

        if conditions['stabilised']['state']:
            eqn += stabilisation_term(alpha=5, zeta=zeta, mesh=mesh, v=self.u1, test=v)
        bcs = DirichletBC(self.V, self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)


class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")
        a = Function(self.U)

        p = TestFunction(self.V)

        h = Constant(1)
        a.interpolate(conditions['ic']['a'])

        ep_dot = self.strain(grad(self.u1))

        zeta = self.zeta(h, a, self.delta(self.u1))
        eta = zeta * params.e ** -2
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * self.Ice_Strength(h,a) * Identity(2)
        eqn = mom_equ(h, self.u1, self.u0, p, sigma, params.rho, uh=self.u0, ocean_curr=conditions['ocean_curr'],
                      rho_a=params.rho_a, C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w,
                      geo_wind=conditions['geo_wind'], cor=params.cor)

        self.u0.assign(conditions['ic']['u'])
        self.u1.assign(self.u0)

        bcs = DirichletBC(self.V, self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)


class ViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W2)
        self.w1 = Function(self.W2)

        u0, h0, a0 = self.w0.split()

        p, q, r = TestFunctions(self.W2)

        u0.assign(conditions['ic']['u'])
        h0.assign(conditions['ic']['h'])
        a0.interpolate(conditions['ic']['a'])

        self.w1.assign(self.w0)

        u1, h1, a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        theta = conditions['theta']
        uh = (1-theta) * u0 + theta * u1
        ah = (1-theta) * a0 + theta * a1
        hh = (1-theta) * h0 + theta * h1

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

        eqn = mom_equ(hh, u1, u0, p, sigma, params.rho, uh=uh, ocean_curr=conditions['ocean_curr'], rho_a=params.rho_a,
                      C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, geo_wind=conditions['geo_wind'],
                      cor=params.cor)
        trans_eqn = trans_equ(0.5, 0.5, uh, hh, ah, h1, h0, a1, a0, q, r, self.n)
        eqn += trans_eqn

        bcs = DirichletBC(self.W2.sub(0), self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()
        
        

class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W1)
        self.w1 = Function(self.W1)
        a = Function(self.U)
        h = Constant(1)

        u0, s0 = self.w0.split()
        p, q = TestFunctions(self.W1)

        u0.assign(conditions['ic']['u'])
        a.interpolate(conditions['ic']['a'])
        s0.assign(conditions['ic']['s'])

        self.w1.assign(self.w0)
        u1, s1 = split(self.w1)
        u0, s0 = split(self.w0)

        theta = conditions['theta']
        uh = (1-theta) * u0 + theta * u1
        sh = (1-theta) * s0 + theta * s1

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(h, a, self.delta(uh))

        if conditions['steady_state']:
            ind = 0
        else:
            ind = 1
        
        eqn = mom_equ(h, u1, u0, p, sh, params.rho, uh=uh, ocean_curr=conditions['ocean_curr'], rho_a=params.rho_a,
                      C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, geo_wind=conditions['geo_wind'],
                      cor=params.cor, ind=ind)
        rheology = params.e ** 2 * sh + Identity(2) * 0.5 * ((1 - params.e ** 2) * tr(sh) + self.Ice_Strength(h, a))
        eqn += inner(q, ind * (s1 - s0) + 0.5 * self.timestep * rheology / params.T) * dx
        eqn -= inner(q * zeta * self.timestep / params.T, ep_dot) * dx

        if conditions['stabilised']['state']:
            alpha = conditions['stabilised']['alpha']
            fix_zeta = self.zeta(alpha, conditions['ic']['u'], params.Delta_min)
            eqn += stabilisation_term(alpha=alpha, zeta=fix_zeta, mesh=mesh, v=uh, test=p)
        bcs = DirichletBC(self.W1.sub(0), self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

        self.u1, self.s1 = self.w1.split()


class ElasticViscousPlasticStress(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        self.sigma0 = Function(self.S, name="StressTensor")
        self.sigma1 = Function(self.S, name="StressTensorNext")

        theta = conditions['theta']
        uh = (1-theta) * u0 + theta * u1

        a = Function(self.U)

        p = TestFunction(self.V)
        q = TestFunction(self.S)

        self.u0.assign(conditions['ic']['u'])
        h = Constant(1)
        a.interpolate(conditions['ic']['a'])

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

        sh = (1-theta) * s + theta * self.sigma0

        eqn = mom_equ(h, self.u1, self.u0, p, sh, params.rho, uh=uh, ocean_curr=conditions['ocean_curr'],
                      rho_a=params.rho_a, C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w,
                      geo_wind=conditions['geo_wind'], cor=params.cor)

        tensor_eqn = inner(self.sigma1 - s, q) * dx

        bcs = DirichletBC(self.V,self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(equ, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)
        sprob = NonlinearVariationalProblem(tensor_eqn, self.sigma1)
        self.ssolver = NonlinearVariationalSolver(sprob, solver_parameters=solver_params.srt_params)


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W3)
        self.w1 = Function(self.W3)

        u0, s0, h0, a0 = self.w0.split()

        p, m, q, r = TestFunctions(self.W3)

        u0.assign(conditions['ic']['u'])
        s0.assign(conditions['ic']['s'])
        h0.assign(conditions['ic']['h'])
        a0.interpolate(conditions['ic']['a'])

        self.w1.assign(self.w0)

        u1, s1, h1, a1 = split(self.w1)
        u0, s0, h0, a0 = split(self.w0)

        theta = conditions['theta']
        uh = (1-theta) * u0 + theta * u1
        sh = (1-theta) * s0 + theta * s1
        hh = (1-theta) * h0 + theta * h1
        ah = (1-theta) * a0 + theta * a1

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(hh, ah, self.delta(uh))
        eta = zeta * params.e ** (-2)

        if  conditions['steady_state']:
            ind = 0
        else:
            ind = 1
            
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


        eqn = mom_equ(hh, u1, u0, p, sh, params.rho, uh=uh, ocean_curr=conditions['ocean_curr'], rho_a=params.rho_a,
                      C_a=params.C_a, rho_w=params.rho_w, C_w=params.C_w, geo_wind=conditions['geo_wind'],
                      cor=params.cor, ind=ind)
        rheology = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - self.Ice_Strength(hh, ah) * 0.5 * Identity(2)
        trans_eqn = trans_equ(0.5, 0.5, uh, hh, ah, h1, h0, a1, a0, q, r, self.n)
        eqn += trans_eqn
        eqn += inner(m, ind * (s1 - s0) + 0.5 * self.timestep * rheology / params.T) * dx
        eqn -= inner(m * zeta * self.timestep / params.T, ep_dot) * dx

        if conditions['stabilised']['state']:
            alpha = conditions['stabilised']['alpha']
            fix_zeta = self.zeta(alpha, conditions['ic']['u'], params.Delta_min)
            eqn += stabilisation_term(alpha=alpha, zeta=fix_zeta, mesh=mesh, v=uh, test=p)

        bcs = DirichletBC(self.W3.sub(0), self.conditions['ic']['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.s0, self.h1, self.a1 = self.w1.split()
