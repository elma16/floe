
class MomentumEquation(object):
    def __init__(self):
        def mom_equ(self, hh, u1, u0, p, sigma, rho, func1=None, uh=None, ocean_curr=None, rho_a=None, C_a=None,
                    rho_w=None,
                    C_w=None, geo_wind=None):
            # TODO make forcing cross product shorter
            def momentum():
                return inner(rho * hh * (u1 - u0), p) * dx

            def forcing():
                if ocean_curr is None:
                    return 0
                else:
                    return inner(rho * hh * cor * as_vector([u1[1] - ocean_curr[1], ocean_curr[0] - u1[0]]), p) * dx

            def stress1(density, drag):
                if ocean_curr is None:
                    return 0
                else:
                    return inner(density * drag * sqrt(dot(ocean_curr - uh, ocean_curr - uh)) * (ocean_curr - uh),
                                 p) * dx

            def stress2(density, drag, func):
                if func is None:
                    return 0
                else:
                    return inner(density * drag * sqrt(dot(func, func)) * func, p) * dx

            def alt_forcing():
                if func1 is None:
                    return 0
                else:
                    return inner(func1, p) * dx

            def rheo():
                return inner(sigma, grad(p)) * dx

            return momentum() + self.timestep * (
                    alt_forcing() - forcing() - stress1(rho_w, C_w) - stress2(rho_a, C_a, geo_wind) - rheo())

class TransportEquation(object):
    def __init__(self):
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
