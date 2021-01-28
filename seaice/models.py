from firedrake import *


class SeaIceModel(object):
    """
    Defining the general class for a Sea Ice Model

    :arg mesh:
    :arg length:
    :arg rheology: The rheology used in the sea ice model.
        'VP' - Viscous-Plastic
        'EVP' - Elastic-Viscous-Plastic
    :arg timestepping:
    :arg number_of_triangles:
    :arg params:
    :output:
    :solver_params:
    """

    def __init__(self, mesh, length, rheology, timestepping, params, output, solver_params):
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
        self.all_u = []
        self.mesh = mesh
        self.length = length
        self.data = {'velocity': []}
        self.rheology = rheology

        timestep = timestepping.timestep
        x, y = SpatialCoordinate(mesh)
        # defining the function spaces
        V = VectorFunctionSpace(mesh, "CR", 1)
        U = FunctionSpace(mesh, "CR", 1)
        W = MixedFunctionSpace([V, U, U])

        self.w0 = Function(W)
        self.w1 = Function(W)

        u0, h0, a0 = self.w0.split()

        # test functions
        p, q, r = TestFunctions(W)

        # TODO initial conditions

        u0.assign(0)
        h0.assign(1)
        a0.assign(1)
        #self.a0.interpolate(x / length)

        self.w1.assign(self.w0)

        self.u1, self.h1, self.a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        uh = 0.5 * (u0 + self.u1)
        ah = 0.5 * (a0 + self.a1)
        hh = 0.5 * (h0 + self.h1)

        def sigma():
            if rheology == 'VP':
                ep_dot = 0.5 * (grad(uh) + transpose(grad(uh)))
                P = params.P_star * hh * exp(-params.C * (1 - ah))
                Delta = sqrt(
                    params.Delta_min ** 2 + 2 * params.e ** (-2) * inner(dev(ep_dot), dev(ep_dot)) + tr(ep_dot) ** 2)
                zeta = 0.5 * P / Delta
                eta = zeta * params.e ** (-2)
                sig = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P * 0.5 * Identity(2)
            elif rheology == 'EVP':
                sig = 1
            return sig

        lm = inner(params.rho * hh * (self.u1 - u0), p) * dx
        # TODO ADD FORCING AND ADVECTION
        lm += timestep * inner(sigma(), grad(p)) * dx

        bcs = [DirichletBC(W.sub(0), 0, "on_boundary")]

        uprob = NonlinearVariationalProblem(lm, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.bt_params)

        self.u1, self.h1, self.a1 = self.w1.split()

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

    def progress(self, t):
        print("Time:", t, "[s]")
        print(int(min(t / self.timescale * 100, 100)), "% complete")
