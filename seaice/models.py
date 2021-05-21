from firedrake import *


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

        if conditions.steady_state == True:
            self.ind = 1
        else:
            self.ind = 1
        
        family = conditions.family
        self.x, self.y = SpatialCoordinate(mesh)
        self.n = FacetNormal(mesh)
        self.V = VectorFunctionSpace(mesh, family, conditions.order + 1)
        self.U = FunctionSpace(mesh, family, conditions.order + 1)
        self.U1 = FunctionSpace(mesh, 'DG', conditions.order)
        self.S = TensorFunctionSpace(mesh, 'DG', conditions.order)
        self.D = FunctionSpace(mesh, 'DG', 0)
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

    def bcs(self, space, location="on_boundary"):
        return [DirichletBC(space, values, location) for values in self.conditions.bc]

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

    def initial_condition(self, *args):
        '''
        arguments should be put in order (variable1, ic1), (variable2, ic2), etc.
        '''
        for vars, ics in args:
            print(vars, ics)
            if isinstance(ics, (int, float)) or type(ics) == 'ufl.tensors.ListTensor':
                vars.assign(ics)
            else:
                vars.interpolate(ics)


    def assemble(self, eqn, func, bcs, params):
        uprob = NonlinearVariationalProblem(eqn, func, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=params)

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")

    def momentum_equation(self, hh, u1, u0, p, sigma, rho, uh, ocean_curr, rho_a, C_a, rho_w, C_w, geo_wind, cor, timestep, ind=1):
        def momentum_term():
            return inner(rho * hh * (u1 - u0), p) * dx

        def forcing_term():
            return inner(rho * hh * cor * perp(ocean_curr - uh), p) * dx

        def stress_term(density, drag, func):
            return inner(density * drag * sqrt(dot(func, func)) * func, p) * dx(degree=3)
    
        def rheology_term():
            return inner(sigma, grad(p)) * dx

        return ind * momentum_term() + timestep * (rheology_term() - forcing_term()
                                                   - stress_term(rho_w, C_w, ocean_curr - uh)
                                                   - stress_term(rho_a, C_a, geo_wind))

    def transport_equation(self, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep):
   
        def in_term(var1, var2, test):
            trial = var2 - var1
            return test * trial * dx

        def upwind_term(var1, test):
            un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
            return timestep * (var1 * div(test * uh) * dx
                               - (test('+') - test('-')) * (un('+') * ah('+') - un('-') * var1('-')) * dS)
    
        return in_term(h0, h1, q) - upwind_term(hh, q) + in_term(a0, a1, r) - upwind_term(ah, r)

    def stabilisation_term(self, alpha, zeta, mesh, v, test):
        e = avg(CellVolume(mesh)) / FacetArea(mesh)
        return 2 * alpha * zeta / e * (dot(jump(v), jump(test))) * dS

class ViscousPlastic(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)
    
        self.u0 = Function(self.V)
        self.u1 = Function(self.V)
        self.h = Function(self.U)
        self.a = Function(self.U)
        
        self.p = TestFunction(self.V)

        theta = conditions.theta
        self.uh = (1-theta) * self.u0 + theta * self.u1

        ep_dot = self.strain(grad(self.uh))
        
        self.initial_condition((self.u0, conditions.ic['u']),(self.u1, self.u0),
                               (self.a, conditions.ic['a']),(self.h, conditions.ic['h']))

        zeta = self.zeta(self.h, self.a, self.delta(self.uh))
        eta = zeta * params.e ** -2
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * self.Ice_Strength(self.h,self.a) * Identity(2)

        self.eqn = self.momentum_equation(self.h, self.u1, self.u0, self.p, sigma, params.rho, self.uh,
                                          conditions.ocean_curr, params.rho_a, params.C_a, params.rho_w,
                                          params.C_w, conditions.geo_wind, params.cor, self.timestep)

        if conditions.stabilised['state']:
            alpha = conditions.stabilised['alpha']
            self.eqn += self.stabilisation_term(alpha=alpha, zeta=avg(zeta), mesh=mesh, v=self.uh, test=self.p)
            
        self.bcs = DirichletBC(self.V, conditions.bc['u'], "on_boundary")

        
class ViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W2)
        self.w1 = Function(self.W2)

        u0, h0, a0 = self.w0.split()

        p, q, r = TestFunctions(self.W2)

        self.initial_condition((u0, conditions.ic['u']), (h0, conditions.ic['h']),
                               (a0, conditions.ic['a']))

        self.w1.assign(self.w0)
        u1, h1, a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        theta = conditions.theta
        uh = (1-theta) * u0 + theta * u1
        ah = (1-theta) * a0 + theta * a1
        hh = (1-theta) * h0 + theta * h1

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(hh, ah, self.delta(uh))
        eta = zeta * params.e ** (-2)
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - self.Ice_Strength(hh, ah) * 0.5 * Identity(
            2)

        eqn = self.momentum_equation(hh, u1, u0, p, sigma, params.rho, uh, conditions.ocean_curr, params.rho_a,
                                params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, self.timestep)
        eqn += self.transport_equation(uh, hh, ah, h1, h0, a1, a0, q, r, self.n, self.timestep)

        if conditions.stabilised['state']:
            alpha = conditions.stabilised['alpha']
            eqn += self.stabilisation_term(alpha=alpha, zeta=avg(zeta), mesh=mesh, v=uh, test=p)

        bcs = DirichletBC(self.W2.sub(0), conditions.bc['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()
        
        

class ElasticViscousPlastic(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W1)
        self.w1 = Function(self.W1)
        self.a = Function(self.U)
        self.h = Function(self.U)

        self.u0, self.s0 = self.w0.split()
        self.p, self.q = TestFunctions(self.W1)

        self.initial_condition((self.u0, conditions.ic['u']), (self.s0, conditions.ic['s']),
                               (self.a, conditions.ic['a']), (self.h, conditions.ic['h']))
        print(norm(self.s0))
        self.w1.assign(self.w0)
        u1, s1 = split(self.w1)
        u0, s0 = split(self.w0)

        theta = conditions.theta
        uh = (1-theta) * u0 + theta * u1
        sh = (1-theta) * s0 + theta * s1

        self.ep_dot = self.strain(grad(uh))
        zeta = self.zeta(self.h, self.a, self.delta(uh))
        self.rheology = params.e ** 2 * sh + Identity(2) * 0.5 * ((1 - params.e ** 2) * tr(sh) + self.Ice_Strength(self.h, self.a))
        
        self.eqn = self.momentum_equation(self.h, u1, u0, self.p, sh, params.rho, uh, conditions.ocean_curr, params.rho_a,
                                          params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, self.timestep, ind=self.ind)
        self.eqn += inner(self.ind * (s1 - s0) + 0.5 * self.timestep * self.rheology / params.T, self.q) * dx
        self.eqn -= inner(self.q * zeta * self.timestep / params.T, self.ep_dot) * dx

        if conditions.stabilised['state']:
            alpha = conditions.stabilised['alpha']
            self.eqn += self.stabilisation_term(alpha=alpha, zeta=avg(zeta), mesh=mesh, v=uh, test=self.p)
            
        self.bcs = DirichletBC(self.W1.sub(0), conditions.bc['u'], "on_boundary")


        #u1, s1 = self.w1.split()


class ElasticViscousPlasticStress(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.u0 = Function(self.V)
        self.u1 = Function(self.V)

        self.sigma0 = Function(self.S)
        self.sigma1 = Function(self.S)

        theta = conditions.theta
        uh = (1-theta) * self.u0 + theta * self.u1

        a = Function(self.U)
        h = Function(self.U)
        
        p = TestFunction(self.V)
        q = TestFunction(self.S)

        self.initial_condition((self.u0, conditions.ic['u']), (a, conditions.ic['a']), (h, conditions.ic['h']))
        
        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(h, a, self.delta(uh))
        eta = zeta * params.e ** (-2)
        rheology = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - 0.5 * self.Ice_Strength(h, a) * Identity(2)

        self.initial_condition((self.sigma0, rheology),(self.sigma1, self.sigma0))

        def sigma_next(timestep, zeta, ep_dot, sigma, P):
            A = 1 + 0.25 * (timestep * params.e ** 2) / params.T
            B = timestep * 0.125 * (1 - params.e ** 2) / params.T
            rhs = (1 - (timestep * params.e ** 2) / (4 * params.T)) * sigma - timestep / params.T * (
                    0.125 * (1 - params.e ** 2) * tr(sigma) * Identity(2) - 0.25 * P * Identity(2) + zeta * ep_dot)
            C = (rhs[0, 0] - rhs[1, 1]) / A
            D = (rhs[0, 0] + rhs[1, 1]) / (A + 2 * B)
            sigma00 = 0.5 * (C + D)
            sigma11 = 0.5 * (D - C)
            sigma01 = rhs[0, 1]
            sigma = as_matrix([[sigma00, sigma01], [sigma01, sigma11]])

            return sigma

        s = sigma_next(self.timestep, zeta, ep_dot, self.sigma0, self.Ice_Strength(h, a))

        sh = (1-theta) * s + theta * self.sigma0

        eqn = self.momentum_equation(h, self.u1, self.u0, p, sh, params.rho, uh, conditions.ocean_curr,
                                params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, self.timestep, ind=self.ind)

        tensor_eqn = inner(self.sigma1-s, q) * dx

        if conditions.stabilised['state']:
            alpha = conditions.stabilised['alpha']
            eqn += stabilisation_term(alpha=alpha, zeta=avg(zeta), mesh=mesh, v=uh, test=p)

        bcs = DirichletBC(self.V, conditions.bc['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.u1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)
        sprob = NonlinearVariationalProblem(tensor_eqn, self.sigma1)
        self.ssolver = NonlinearVariationalSolver(sprob, solver_parameters=solver_params.bt_params)


class ElasticViscousPlasticTransport(SeaIceModel):
    def __init__(self, mesh, conditions, timestepping, params, output, solver_params):
        super().__init__(mesh, conditions, timestepping, params, output, solver_params)

        self.w0 = Function(self.W3)
        self.w1 = Function(self.W3)

        u0, s0, h0, a0 = self.w0.split()

        p, q, r, m = TestFunctions(self.W3)

        self.initial_condition((u0, conditions.ic['u']), (s0, conditions.ic['s']),
                               (a0, conditions.ic['a']), (h0, conditions.ic['h']))

        self.w1.assign(self.w0)

        u1, s1, h1, a1 = split(self.w1)
        u0, s0, h0, a0 = split(self.w0)

        theta = conditions.theta
        uh = (1-theta) * u0 + theta * u1
        sh = (1-theta) * s0 + theta * s1
        hh = (1-theta) * h0 + theta * h1
        ah = (1-theta) * a0 + theta * a1

        ep_dot = self.strain(grad(uh))
        zeta = self.zeta(hh, ah, self.delta(uh))

        rheology = params.e ** 2 * sh + Identity(2) * 0.5 * ((1 - params.e ** 2) * tr(sh) + self.Ice_Strength(hh, ah))
        
        eqn = self.momentum_equation(hh, u1, u0, p, sh, params.rho, uh, conditions.ocean_curr, params.rho_a,
                                params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, self.timestep, ind=self.ind)
        eqn += self.transport_equation(uh, hh, ah, h1, h0, a1, a0, r, m, self.n, self.timestep)
        eqn += inner(self.ind * (s1 - s0) + 0.5 * self.timestep * rheology / params.T, q) * dx
        eqn -= inner(q * zeta * self.timestep / params.T, ep_dot) * dx

        if conditions.stabilised['state']:
            alpha = conditions.stabilised['alpha']
            eqn += self.stabilisation_term(alpha=alpha, zeta=avg(zeta), mesh=mesh, v=uh, test=p)

        bcs = DirichletBC(self.W3.sub(0), conditions.bc['u'], "on_boundary")

        uprob = NonlinearVariationalProblem(eqn, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.s0, self.h1, self.a1 = self.w1.split()
