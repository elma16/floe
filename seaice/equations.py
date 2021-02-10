from firedrake import *

class Equation(object):
    def __init__(self, h_in, a_in, uh, hh, ah, h1, h0, a1, a0, q, r, n, timestep, u1, u0, sigma, rho, p, cor,
                 ocean_curr, rho_w, rho_a=None, C_w=None, C_a=None, geo_wind=None):
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
        self.timestep = timestep

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

    def neglect(self, term):
        if None in locals().values():
            return 0
        else:
            pass

    def assemble(self):
        return self.mom_equ(self.hh, self.u1, self.u0, self.p, self.sigma, self.rho, func1=None, uh=None, ocean_curr=None, rho_a=None, C_a=None, rho_w=None,
                C_w=None, geo_wind=None)

