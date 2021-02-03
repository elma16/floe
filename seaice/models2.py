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

    def __init__(self, mesh, bcs_values, ics_values, length, rheology, timestepping, params, output, solver_params):
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
        self.rheology = rheology
        self.bcs_values = bcs_values
        self.ics_values = ics_values

        timestep = timestepping.timestep
        x, y = SpatialCoordinate(mesh)

        V = VectorFunctionSpace(mesh, "CR", 1)
        U = FunctionSpace(mesh, "CR", 1)
        W = MixedFunctionSpace([V, U, U])

        self.w0 = Function(W)
        self.w1 = Function(W)

        u0, h0, a0 = self.w0.split()

        p, q, r = TestFunctions(W)

        u0.assign(ics_values[0])
        h0.assign(ics_values[1])
        a0.assign(ics_values[2])

        self.w1.assign(self.w0)

        u1, h1, a1 = split(self.w1)
        u0, h0, a0 = split(self.w0)

        uh = 0.5 * (u0 + u1)
        ah = 0.5 * (a0 + a1)
        hh = 0.5 * (h0 + h1)

        P = params.P_star * hh * exp(-params.C * (1 - ah))
        zeta = 0.5 * P / params.Delta_min

        sigma = 0.5 * zeta * (grad(uh) + transpose(grad(uh)))
        pi_x = pi / length
        v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
        sigma_exp = 0.5 * zeta * (grad(v_exp) + transpose(grad(v_exp)))
        R = -div(sigma_exp)

        def strain(omega):
            return 0.5 * (omega + transpose(omega))

        # TODO ADD FORCING AND ADVECTION
        lm = inner(hh*(u1 - u0), p) * dx
        lm += timestep * inner(sigma, strain(grad(p))) * dx
        lm -= timestep * inner(R, p) * dx

        bcs = [DirichletBC(W.sub(bcs_values.index(values)), values, "on_boundary") for values in bcs_values]

        uprob = NonlinearVariationalProblem(lm, self.w1, bcs)
        self.usolver = NonlinearVariationalSolver(uprob, solver_parameters=solver_params.srt_params)

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
