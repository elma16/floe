import inspect
import os
import sys

from firedrake import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


# TODO : get component of UFL velocity


class SeaIceModel(object):
    def __init__(self, timescale, timestep, number_of_triangles, params, output):
        self.timescale = timescale
        self.timestep = timestep
        self.number_of_triangles = number_of_triangles
        self.params = params
        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output
        self.outfile = File(output.dirname)
        self.dump_count = 0
        self.dump_freq = output.dumpfreq

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.length)

    # TODO get some shared methods into here


class StrainRateTensor(SeaIceModel):
    def __init__(self, stabilised, transform_mesh, output, shape, params, timescale, timestep, number_of_triangles):

        """
        Given the initial conditions, create the equations with the variables given

        TODO add stabilised, transform mesh, shape
        """
        super().__init__(timescale, timestep, number_of_triangles, params, output)
        self.stabilised = stabilised
        self.shape = shape
        self.transform_mesh = transform_mesh

        if output is None:
            raise RuntimeError("You must provide a directory name for dumping results")
        else:
            self.output = output

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)

        # sea ice velocity
        self.u0 = Function(self.V, name="Velocity")
        self.u1 = Function(self.V, name="VelocityNext")

        # test functions
        self.v = TestFunction(self.V)

        x, y = SpatialCoordinate(self.mesh)

        self.h = Constant(1)
        self.a = Constant(1)

        # ice strength
        P = params.P_star * self.h * exp(-params.C * (1 - self.a))

        # viscosities
        zeta = 0.5 * P / params.Delta_min

        sigma = 0.5 * zeta * (grad(self.u1) + transpose(grad(self.u1)))

        pi_x = pi / params.length
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))

        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        self.bcs = [DirichletBC(self.V, Constant(0), "on_boundary")]

        # momentum equation
        self.lm = (inner(self.u1 - self.u0, self.v) + timestep * inner(sigma, strain(grad(self.v)))) * dx
        self.lm -= timestep * inner(R, self.v) * dx

        solver_params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu"}

        self.uprob = NonlinearVariationalProblem(self.lm, self.u1, self.bcs)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters=solver_params)

    def solve(self, t):

        """
        Solve the equations at a given timestep
        """
        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):
        """
        Update the equations with the new values of the functions
        """

        while t < self.timescale - 0.5 * self.timestep:
            StrainRateTensor.solve(self, t)
            self.u0.assign(self.u1)
            t += self.timestep

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)


class Evp(SeaIceModel):
    def __init__(self, timescale, timestep, number_of_triangles, params, output):

        """
        Given the initial conditions, create the equations with the variables given

        """
        super().__init__(timescale, timestep, number_of_triangles, params, output)

    def solve(self, t):

        """
        Solve the equations at a given timestep
        """
        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):
        """
        Update the equations with the new values of the functions
        """

        while t < self.timescale - 0.5 * self.timestep:
            Evp.solve(self, t)
            self.u0.assign(self.u1)
            t += self.timestep

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)


class BoxTest(SeaIceModel):
    def __init__(self, timescale, timestep, number_of_triangles, stabilised, output, params):
        """
        Given the initial conditions, output the equations
        """
        super().__init__(timescale, timestep, number_of_triangles, params, output)

        self.mesh = SquareMesh(number_of_triangles, number_of_triangles, params.box_length)

        self.V = VectorFunctionSpace(self.mesh, "CR", 1)
        self.U = FunctionSpace(self.mesh, "CR", 1)
        self.W = MixedFunctionSpace([self.V, self.U, self.U])

        self.w0 = Function(self.W)
        self.w1 = Function(self.W)

        self.u0, self.h0, self.a0 = self.w0.split()

        # test functions
        p, q, r = TestFunctions(self.W)

        x, y = SpatialCoordinate(mesh)

        # initial conditions
        self.u0.assign(0)
        self.h0.assign(1)
        self.a0.interpolate(x / params.box_length)

        self.w1.assign(self.w0)

        self.u1, self.h1, self.a1 = split(self.w1)
        self.u0, self.h0, self.a0 = split(self.w0)

        self.uh = 0.5 * (self.u0 + self.u1)
        self.ah = 0.5 * (self.a0 + self.a1)
        self.hh = 0.5 * (self.h0 + self.h1)

        # boundary condition
        h_in = Constant(0.5)
        a_in = Constant(0.5)

        # ocean current
        ocean_curr = as_vector([0.1 * (2 * y - params.box_length) / params.box_length,
                                -0.1 * (params.box_length - 2 * x) / params.box_length])

        # strain rate tensor
        ep_dot = 0.5 * (grad(self.uh) + transpose(grad(self.uh)))

        # ice strength
        P = params.P_star * self.hh * exp(-params.C * (1 - self.ah))

        Delta = sqrt(params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)

        if stabilised == 0:
            stab_term = 0
        if stabilised == 1:
            stab_term = 2 * params.a_vp * avg(CellVolume(self.mesh)) / FacetArea(self.mesh) * (
                dot(jump(self.u1), jump(p))) * dS

        # viscosities
        zeta = 0.5 * P / Delta
        eta = zeta * params.e ** (-2)

        # internal stress tensor
        sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)

        # initalise geo_wind
        t0 = Constant(0)

        geo_wind = as_vector([5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * x / params.box_length) * sin(
            2 * pi * y / params.box_length),
                              5 + (sin(2 * pi * t0 / timescale) - 3) * sin(2 * pi * y / params.box_length) * sin(
                                  2 * pi * x / params.box_length)])

        lm = inner(params.rho * self.hh * (self.u1 - self.u0), p) * dx
        lm -= timestep * inner(params.rho * self.hh * params.cor * as_vector([self.uh[1] - ocean_curr[1], ocean_curr[0]
                                                                              - self.uh[0]]), p) * dx
        lm += timestep * inner(
            params.rho_a * params.C_a * dot(geo_wind, geo_wind) * geo_wind + params.rho_w * params.C_w * sqrt(
                dot(self.uh - ocean_curr, self.uh - ocean_curr)) * (
                    ocean_curr - self.uh), p) * dx
        lm += timestep * inner(sigma, grad(p)) * dx
        lm += stab_term

        # adding the transport equations
        dh_trial = self.h1 - self.h0
        da_trial = self.a1 - self.a0

        # LHS
        lm += q * dh_trial * dx
        lm += r * da_trial * dx

        self.n = FacetNormal(self.mesh)

        un = 0.5 * (dot(self.uh, self.n) + abs(dot(self.uh, self.n)))

        lm -= timestep * (self.hh * div(q * self.uh) * dx
                          - conditional(dot(self.uh, self.n) < 0, q * dot(self.uh, self.n) * h_in, 0.0) * ds
                          - conditional(dot(self.uh, self.n) > 0, q * dot(self.uh, self.n) * self.hh, 0.0) * ds
                          - (q('+') - q('-')) * (un('+') * self.ah('+') - un('-') * self.hh('-')) * dS)

        lm -= timestep * (self.ah * div(r * self.uh) * dx
                          - conditional(dot(self.uh, self.n) < 0, r * dot(self.uh, self.n) * a_in, 0.0) * ds
                          - conditional(dot(self.uh, self.n) > 0, r * dot(self.uh, self.n) * self.ah, 0.0) * ds
                          - (r('+') - r('-')) * (un('+') * self.ah('+') - un('-') * self.ah('-')) * dS)

        bcs = [DirichletBC(self.W.sub(0), 0, "on_boundary")]

        params2 = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu", 'mat_type': 'aij'}
        self.uprob = NonlinearVariationalProblem(lm, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters=params2)

        self.u1, self.h1, self.a1 = self.w1.split()

    def solve(self, t):

        self.usolver.solve()

        if t == 0:
            self.outfile.write(self.u1, time=t)

    def update(self, t):

        while t < self.timescale - 0.5 * self.timestep:
            BoxTest.solve(self, t)
            self.w0.assign(self.w1)
            t += self.timestep

    def dump(self, t):
        """
        Output the diagnostics
        """
        self.dump_count += 1
        if self.dump_count == self.dump_freq:
            self.dump_count -= self.dump_freq
            self.outfile.write(self.u1, time=t)
