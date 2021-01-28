class Advection(object):
    def __init__(self, ):
        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        # adding the transport equations
        dh_trial = self.h1 - self.h0
        da_trial = self.a1 - self.a0

        # LHS
        self.lm += q * dh_trial * dx
        self.lm += r * da_trial * dx

        self.n = FacetNormal(mesh)

        un = 0.5 * (dot(self.uh, self.n) + abs(dot(self.uh, self.n)))

        self.lm -= self.timestep * (self.hh * div(q * self.uh) * dx
                                    - conditional(dot(self.uh, self.n) < 0, q * dot(self.uh, self.n) * h_in, 0.0) * ds
                                    - conditional(dot(self.uh, self.n) > 0, q * dot(self.uh, self.n) * self.hh,
                                                  0.0) * ds
                                    - (q('+') - q('-')) * (un('+') * self.ah('+') - un('-') * self.hh('-')) * dS)

        self.lm -= self.timestep * (self.ah * div(r * self.uh) * dx
                                    - conditional(dot(self.uh, self.n) < 0, r * dot(self.uh, self.n) * a_in, 0.0) * ds
                                    - conditional(dot(self.uh, self.n) > 0, r * dot(self.uh, self.n) * self.ah,
                                                  0.0) * ds
                                    - (r('+') - r('-')) * (un('+') * self.ah('+') - un('-') * self.ah('-')) * dS)


