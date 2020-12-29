import inspect
import os
import sys
from firedrake import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

from seaice.models import *

# TODO Get the Error diagnostic to the point in which you can plot stuff with it

timestepping = TimesteppingParameters(timescale=10, timestep=10 ** (-1))
params = SeaIceParameters()


class Diagnostic(object):
    def __init__(self, model):
        if not isinstance(model, SeaIceModel):
            raise RuntimeError("You must use a sea ice model")
        else:
            self.model = model


class Error(Diagnostic):
    """
Convergence plots for the strain rate tensor test:
    u vs. t (timestep variable)
    u vs. t (meshsize variable)
    u vs. t (stabilised vs. unstabilised)
    """

    def __init__(self, dirname, yaxis, model):
        super().__init__(model)
        self.timescale = timestepping.timescale
        self.timestep = timestepping.timestep
        self.dirname = dirname
        self.yaxis = yaxis

    def compute(self):
        all_u, mesh, v_exp, zeta = self.model
        return [errornorm(v_exp, all_u[i]) for i in range(len(all_u) - 1)]

    def compute2(self):
        """
        Compute the energy of the solution in the instance of the EVP/VP model
        u1 - energy defined pg 8, after energy proof
        u2 - energy defined pg 19
        u3 - energy used on the y axis of the energy plot fig 7, pg 20
        """
        # all_u, all_h, all_a, mesh, zeta = vp_evp_test_explicit(timescale, timestep, number_of_triangles, rheology,
        #                                                       advection=True, solver=solver, subcycle=5,
        #                                                       stabilised=stabilised)

        eta = zeta * e ** (-2)
        energy_u1 = [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u))]
        energy_u2 = [norm(sqrt(zeta) * all_u[i]) for i in range(len(all_u))]
        energy_u3 = [norm(sqrt(eta) * grad(all_u[i])) for i in range(len(all_u))]

        # energy_h = [norm(0.5 * zeta * grad(all_h[i])) for i in range(len(all_h))]
        # energy_a = [norm(0.5 * zeta * grad(all_a[i])) for i in range(len(all_a))]

        return energy_u1, energy_u2, energy_u3

    def plot(self, yaxis):
        t = np.arange(0, self.timescale, self.timestep)
        plt.semilogy(t, compute(), label="timescale = %s" % k)
        plt.ylabel(r'Error of solution ')
        plt.xlabel(r'{}'.format(self.yaxis))
        plt.title(r'Error of computed solution for Section {} Test, ')
        plt.legend(loc='best')
        plt.show()

        plt.savefig(self.dirname)


class Energy(Diagnostic):
    def __init__(self, model):
        super().__init__(model)

    def compute(self, timescale, timestep, number_of_triangles):
        all_u, mesh, v_exp, zeta = self.model(timescale, timestep, number_of_triangles)
        return [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u) - 1)]

    def plot(self):
        # interesting plots:
        # plot_u_conv(2 * 10 ** (-2), 10 ** (-3))
        t = np.arange(0, timescale, timestep)
        plt.semilogy(t, energy(timescale, timestep), label="timescale = %s" % k)
        plt.ylabel(r'Energy of solution ')
        plt.xlabel(r'Time [s]')
        plt.title(r'Energy of computed solution for Section 4.1 Test, k = {}, T = {}'.format(timestep, timescale))
        plt.legend(loc='best')
        plt.show()
        plt.savefig('./plots/strain_rate_energy.png')


class Velocity(Diagnostic):
    def __init__(self, model):
        super().__init__(model)

    def X_component(self):
        return 0

    def Y_component(self):
        return 0

    def Max_component(self):
        all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
        # projecting the solutions of the problem onto 'DG1'
        W = VectorFunctionSpace(mesh, "DG", 1)
        p = [project(all_u[i], W).dat.data for i in range(len(all_u))]
        print(shape(p[0]))
        # print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])
        return all_u


def korn_ineq(timescale, timestep, number_of_triangles=35, stabilised=0):
    """Illustrating the failure of CR1 in Korn's Inequality"""
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    print([norm(grad(all_u[i])) > sqrt(norm(grad(all_u[i]) + transpose(grad(all_u[i])))) for i in range(len(all_u))])
