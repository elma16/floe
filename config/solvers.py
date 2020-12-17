from firedrake import (split, LinearVariationalProblem, Constant,
                       LinearVariationalSolver, TestFunctions, TrialFunctions,
                       TestFunction, TrialFunction, lhs, rhs, FacetNormal,
                       div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, ds_tb, inner,
                       dot, grad, Function, VectorSpaceBasis, BrokenElement,
                       FunctionSpace, MixedFunctionSpace)
from firedrake.petsc import flatten_parameters
from firedrake.parloops import par_loop, READ, INC
from pyop2.profiling import timed_function, timed_region
from abc import ABCMeta, abstractmethod, property


class TimesteppingSolver(object, metaclass=ABCMeta):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class.

    :arg state: :class:`.State` object.
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then update
         the default solver parameters with the solver_parameters passed in.
    """

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.state = state

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        if logger.isEnabledFor(DEBUG):
            self.solver_parameters["ksp_monitor_true_residual"] = None

        # setup the solver
        self._setup_solver()

    @property
    def solver_parameters(self):
        """Solver parameters for this solver"""
        pass

    @abstractmethod
    def solve(self):
        pass


class ForwardEulerSolver(TimesteppingSolver):
    solver_parameters = {"ksp_monitor": None,
                         "snes_monitor": None,
                         "ksp_type": "preonly",
                         "pc_type": "lu"}

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        import numpy as np

        state = self.state
        Dt = state.timestepping.dt
        beta_ = Dt * state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vu_broken = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        dt = Constant(Dt)
        beta = Constant(beta_)
        beta_cp = Constant(beta_ * cp)

        h_deg = state.horizontal_degree
        v_deg = state.vertical_degree
        Vtrace = FunctionSpace(state.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the function space for "broken" u, rho, and pressure trace
        M = MixedFunctionSpace((Vu_broken, Vrho, Vtrace))
        w, phi, dl = TestFunctions(M)
        u, rho, l0 = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = state.k  # Upward pointing unit vector
        theta = -dot(k, u) * dot(k, grad(thetabar)) * beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # The pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        pi = pibar_theta * theta + pibar_rho * rho

        # Vertical projection
        def V(u):
            return k * inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = (ds_t(degree=(self.quadrature_degree))
                  + ds_b(degree=(self.quadrature_degree)))


        theta_w = theta
        thetabar_w = thetabar

        _l0 = TrialFunction(Vtrace)
        _dl = TestFunction(Vtrace)
        a_tr = _dl('+') * _l0('+') * (dS_vp + dS_hp) + _dl * _l0 * ds_vp + _dl * _l0 * ds_tbp

        def L_tr(f):
            return _dl('+') * avg(f) * (dS_vp + dS_hp) + _dl * f * ds_vp + _dl * f * ds_tbp

        cg_ilu_parameters = {'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        # Project field averages into functions on the trace space
        rhobar_avg = Function(Vtrace)
        pibar_avg = Function(Vtrace)

        rho_avg_prb = LinearVariationalProblem(a_tr, L_tr(rhobar), rhobar_avg)
        pi_avg_prb = LinearVariationalProblem(a_tr, L_tr(pibar), pibar_avg)

        rho_avg_solver = LinearVariationalSolver(rho_avg_prb,
                                                 solver_parameters=cg_ilu_parameters,
                                                 options_prefix='rhobar_avg_solver')
        pi_avg_solver = LinearVariationalSolver(pi_avg_prb,
                                                solver_parameters=cg_ilu_parameters,
                                                options_prefix='pibar_avg_solver')

        with timed_region("Gusto:HybridProjectRhobar"):
            rho_avg_solver.solve()

        with timed_region("Gusto:HybridProjectPibar"):
            pi_avg_solver.solve()

        # "broken" u, rho, and trace system
        # NOTE: no ds_v integrals since equations are defined on
        # a periodic (or sphere) base mesh.
        eqn = (
            # momentum equation
                inner(w, (state.h_project(u) - u_in)) * dx
                - beta_cp * div(theta_w * V(w)) * pibar * dxp
                # following does nothing but is preserved in the comments
                # to remind us why (because V(w) is purely vertical).
                # + beta_cp*jump(theta_w*V(w), n=n)*pibar_avg('+')*dS_vp
                + beta_cp * jump(theta_w * V(w), n=n) * pibar_avg('+') * dS_hp
                + beta_cp * dot(theta_w * V(w), n) * pibar_avg * ds_tbp
                - beta_cp * div(thetabar_w * w) * pi * dxp
                # trace terms appearing after integrating momentum equation
                + beta_cp * jump(thetabar_w * w, n=n) * l0('+') * (dS_vp + dS_hp)
                + beta_cp * dot(thetabar_w * w, n) * l0 * (ds_tbp + ds_vp)
                # mass continuity equation
                + (phi * (rho - rho_in) - beta * inner(grad(phi), u) * rhobar) * dx
                + beta * jump(phi * u, n=n) * rhobar_avg('+') * (dS_v + dS_h)
                # term added because u.n=0 is enforced weakly via the traces
                + beta * phi * dot(u, n) * rhobar_avg * (ds_tb + ds_v)
                # constraint equation to enforce continuity of the velocity
                # through the interior facets and weakly impose the no-slip
                # condition
                + dl('+') * jump(u, n=n) * (dS_vp + dS_hp)
                + dl * dot(u, n) * (ds_tbp + ds_vp)
        )

        # contribution of the sponge term
        if mu is not None:
            eqn += dt * mu * inner(w, k) * inner(u, k) * dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Function for the hybridized solutions
        self.urhol0 = Function(M)

        hybridized_prb = LinearVariationalProblem(aeqn, Leqn, self.urhol0)
        hybridized_solver = LinearVariationalSolver(hybridized_prb,
                                                    solver_parameters=self.solver_parameters,
                                                    options_prefix='ImplicitSolver')
        self.hybridized_solver = hybridized_solver

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        shapes = {"i": Vu.finat_element.space_dimension(),
                  "j": np.prod(Vu.shape, dtype=int)}
        weight_kernel = """
        for (int i=0; i<{i}; ++i)
            for (int j=0; j<{j}; ++j)
                w[i*{j} + j] += 1.0;
        """.format(**shapes)

        self._weight = Function(Vu)
        par_loop(weight_kernel, dx, {"w": (self._weight, INC)})

        # Averaging kernel
        self._average_kernel = """
        for (int i=0; i<{i}; ++i)
            for (int j=0; j<{j}; ++j)
                vec_out[i*{j} + j] += vec_in[i*{j} + j]/w[i*{j} + j];
        """.format(**shapes)

        # HDiv-conforming velocity
        self.u_hdiv = Function(Vu)

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)
        theta_eqn = gamma * (theta - theta_in
                             + dot(k, self.u_hdiv) * dot(k, grad(thetabar)) * beta) * dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters=cg_ilu_parameters,
                                                    options_prefix='thetabacksubstitution')

        # Store boundary conditions for the div-conforming velocity to apply
        # post-solve
        self.bcs = self.state.bcs

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        # Solve the hybridized system
        self.hybridized_solver.solve()

        broken_u, rho1, _ = self.urhol0.split()
        u1 = self.u_hdiv

        # Project broken_u into the HDiv space
        u1.assign(0.0)

        with timed_region("Gusto:HybridProjectHDiv"):
            par_loop(self._average_kernel, dx,
                     {"w": (self._weight, READ),
                      "vec_in": (broken_u, READ),
                      "vec_out": (u1, INC)})

        # Reapply bcs to ensure they are satisfied
        for bc in self.bcs:
            bc.apply(u1)

        # Copy back into u and rho cpts of dy
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        # Reconstruct theta
        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)
