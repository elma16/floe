from firedrake import *


class Equation(object):
    def __init__(self, h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep):
        self.h_in = h_in
        self.a_in = a_in
        self.uh = uh
        self.hh = hh
        self.ah = ah
        self.h1 = h1
        self.h0 = h0
        self.a1 = a1
        self.a0 = a0
        self.q = q
        self.r = r
        self.n = n
        self.timestep = timestep

    def neglect(self, term):
        if None in locals().values():
            return 0
        else:
            pass

    def assemble(self):
        equ = 0
        for term in Equation.__dict__.values():
            equ += term()
        return equ


class MomentumEquation(Equation):
    def __init__(self, u1, u0, hh, uh, p, sigma, rho, h_in, a_in, ah, h1, h0, a1, a0, q, r, n, timestep, cor,
                 ocean_curr, rho_w, rho_a, C_w, C_a, geo_wind):
        super().__init__(h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep)
        self.u1 = u1
        self.u0 = u0
        self.sigma = sigma
        self.rho = rho
        self.p = p
        self.cor = cor
        self.ocean_curr = ocean_curr
        self.rho_w = rho_w
        self.C_w = C_w
        self.C_a = C_a
        self.rho_a = rho_a
        self.geo_wind = geo_wind

    # TODO make forcing cross product shorter
    def momentum(self):
        return inner(self.rho * self.hh * (self.u1 - self.u0), self.p) * dx

    def forcing(self):
        return inner(self.rho * self.hh * self.cor * as_vector(
            [self.u1[1] - self.ocean_curr[1], self.ocean_curr[0] - self.u1[0]]), self.p) * dx

    def ocean_stress(self):
        return inner(self.rho_w * self.C_w * sqrt(dot(self.ocean_curr - self.uh, self.ocean_curr - self.uh)) * (
                self.ocean_curr - self.uh), self.p) * dx

    def wind_stress(self):
        return inner(self.rho_a * self.C_a * sqrt(dot(self.geo_wind, self.geo_wind)) * self.geo_wind, self.p) * dx

    def rheo(self):
        return inner(self.sigma, grad(self.p)) * dx


class TransportEquation(Equation):
    def __init__(self, h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep):
        super().__init__(h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep)

    def in_term(self, var1, var2, test):
        trial = var2 - var1
        return test * trial * dx

    def upwind_term(self, var1, bc_in, test):
        un = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
        return self.timestep * (var1 * div(test * uh) * dx
                                - conditional(dot(uh, n) < 0, test * dot(uh, n) * bc_in, 0.0) * ds
                                - conditional(dot(uh, n) > 0, test * dot(uh, n) * var1, 0.0) * ds
                                - (test('+') - test('-')) * (un('+') * ah('+') - un('-') * var1('-')) * dS)

    def assemble(self):
        return self.in_term(self.h0, self.h1, self.q) + self.in_term(self.a0, self.a1, self.r) + self.upwind_term(
            self.hh, self.h_in, self.q) + self.upwind_term(self.ah, self.a_in, self.r)


x = Equation(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

functions = sorted([
    getattr(x, field) for field in dir(x)
    if hasattr(getattr(x, field), "order")
],
    key=(lambda field: field.order)
)
for func in functions:
    func()
