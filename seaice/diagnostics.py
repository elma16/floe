from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from seaice.models import *

# TODO Get the Error diagnostic to the point in which you can plot stuff with it
# TODO : get component of UFL velocity


__all__ = ["Error", "Energy", "Velocity"]


class Diagnostic(object):
    def __init__(self, model, dirname, timestepping):
        if not isinstance(model, SeaIceModel):
            raise RuntimeError("You must use a sea ice model")
        else:
            self.model = model
        self.dirname = dirname
        self.timestepping = timestepping
        self.timescale = timestepping.timescale
        self.timestep = timestepping.timestep


class Error(Diagnostic):
    """
Convergence plots for the strain rate tensor test:
    u vs. t (timestep variable)
    u vs. t (meshsize variable)
    u vs. t (stabilised vs. unstabilised)
    """

    def __init__(self, model, dirname, xaxis, timestepping, values):
        super().__init__(model, dirname, timestepping)
        self.xaxis = xaxis
        self.values = values

    @staticmethod
    def compute(model):
        all_u, mesh, v_exp, zeta = model.sp_output()
        return [errornorm(v_exp, all_u[i]) for i in range(len(all_u) - 1)]

    def plot(self, model, xaxis, values):
        for k in values:
            t = np.arange(0, self.timescale, self.timestep)
            plt.semilogy(t, Error.compute(model(k)), label="{} = {}".format(xaxis, values))
            plt.ylabel(r'Error of solution ')
            plt.xlabel(r'{}'.format(xaxis))
            plt.title(r'Error of computed solution for Section {} Test, ')
            plt.legend(loc='best')
        plt.show()

        plt.savefig(self.dirname)


class Energy(Diagnostic):
    def __init__(self, model, dirname, timestepping, params):
        super().__init__(model, dirname, timestepping)

        self.params = params

    def compute(self, model, params):
        """
        Compute the energy of the solution in the instance of the EVP/VP model
        u1 - energy defined pg 8, after energy proof
        u2 - energy defined pg 19
        u3 - energy used on the y axis of the energy plot fig 7, pg 20
        """
        all_u, mesh, v_exp, zeta = model.sp_output()

        eta = zeta * params.e ** (-2)

        # TODO change energy to (v2 of paper) -> integral from 0 to timescale zeta_min norm(grad(v))

        energy_u1 = [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u))]
        energy_u2 = [norm(sqrt(zeta) * all_u[i]) for i in range(len(all_u))]
        energy_u3 = [norm(sqrt(eta) * grad(all_u[i])) for i in range(len(all_u))]

        return energy_u1

    def plot(self, model, params):
        t = np.arange(0, self.timescale, self.timestep)
        plt.semilogy(t, Energy.compute(self, model, params))
        plt.ylabel(r'Energy of solution :{} ')
        plt.xlabel(r'Time [s]')
        plt.title(
            r'Energy of computed solution for Section 4.1 Test, t = {}, T = {}'.format(self.timestep,
                                                                                       self.timescale))
        plt.show()
        plt.savefig(self.dirname)


class Velocity(Diagnostic):
    def __init__(self, model, dirname, timestepping):
        super().__init__(model, dirname, timestepping)

    def X_component(self):
        return 0

    def Y_component(self):
        return 0

    def Max_component(self):
        all_u, mesh, v_exp, zeta = model.sp_output()
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