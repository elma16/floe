import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

from seaice.models import *

'''
Convergence plots for the strain rate tensor test:
    u vs. t (timestep variable)
    u vs. t (meshsize variable)
    u vs. t (stabilised vs. unstabilised)
'''

timestepping = TimesteppingParameters(timescale=10, timestep=10 ** (-1))
params = SeaIceParameters()


class PlotError(object):
    """
    Compute the error and plot it against time
    """

    def __init__(self, dirname, yaxis):
        self.timescale = timestepping.timescale
        self.timestep = timestepping.timestep
        self.dirname = dirname
        self.yaxis = yaxis

    def compute_error(self):
        all_u, mesh, v_exp, zeta = StrainRateTensor(timestepping=timestepping, params=params)
        return [errornorm(v_exp, all_u[i]) for i in range(len(all_u) - 1)]

    def plot(self, yaxis):
        t = np.arange(0, self.timescale, self.timestep)
        plt.semilogy(t, compute_error(), label="timescale = %s" % k)
        plt.ylabel(r'Error of solution ')
        plt.xlabel(r'{}'.format(self.yaxis))
        plt.title(r'Error of computed solution for Section {} Test, ')
        plt.legend(loc='best')
        plt.show()

        plt.savefig(self.dirname)


def plot_u_conv(timescale, timestep, number_of_triangles):
    # interesting plots:
    # plot_u_conv(2 * 10 ** (-2), 10 ** (-3))
    t = np.arange(0, timescale, timestep)
    plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles), label="timescale = %s" % k)
    plt.ylabel(r'Error of solution ')
    plt.xlabel(r'Time [s]')
    plt.title(r'Error of computed solution for Section 4.1 Test, k = {}, T = {}'.format(timestep, timescale))
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_conv.png')


def plot_u_conv_vs_meshsize(timescale, timestep):
    # plotting convergence of velocity with timescale,timestep fixed, and mesh size changing.
    # plot_u_conv_vs_meshsize(4*10**(-1),10**(-2))
    for N in [10, 20, 30, 50, 100, 200]:
        t = np.arange(0, timescale, timestep)
        plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles=N), label='mesh = %s' % N)
        plt.ylabel(r'Error of solution')
        plt.xlabel(r'Time [s]')
        plt.title(r'Error of computed solution for Section 4.1 Test')
        plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_veloN')


def plot_u_conv_vs_stab(timescale, timestep):
    # INCOMPLETE: plotting convergence of velocity with stability changing
    t = np.arange(0, timescale, timestep)
    plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles=100, stabilised=0), 'r--',
                 label=r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
    plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles=10, stabilised=0), 'b.',
                 label=r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
    plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles=100, stabilised=2), 'g--',
                 label=r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v)$')
    plt.semilogy(t, strain_rate_tensor_error(timescale, timestep, number_of_triangles=10, stabilised=2), 'k.',
                 label=r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v)$')
    plt.legend(loc='best')
    plt.show()


class PlotEnergy(object):
    def __init__(self, model):
        self.model = model

    def compute_energy(self, timescale, timestep, number_of_triangles):
        all_u, mesh, v_exp, zeta = self.model(timescale, timestep, number_of_triangles)
        return [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u) - 1)]

    def plot_energy(self):
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


def korn_ineq(timescale, timestep, number_of_triangles=35, stabilised=0):
    """Illustrating the failure of CR1 in Korn's Inequality"""
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    print([norm(grad(all_u[i])) > sqrt(norm(grad(all_u[i]) + transpose(grad(all_u[i])))) for i in range(len(all_u))])



def vel_comp_max(timescale, timestep, number_of_triangles=35, stabilised=0):
    """
    Computing the maximum component of all the velocities in the velocity field
    """
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    # projecting the solutions of the problem onto 'DG1'
    W = VectorFunctionSpace(mesh, "DG", 1)
    p = [project(all_u[i], W).dat.data for i in range(len(all_u))]
    print(shape(p[0]))
    # print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])
    return all_u


vel_comp_max(10, 1)


# EVP

def vp_error(timescale, timestep, number_of_triangles=30, stabilised=0):
    """
    Compute the error norm of the velocity against the stationary solution for test 1
    """
    all_u, mesh, v_exp, zeta = vp_evp_test_explicit(timescale, timestep, number_of_triangles, stabilised)
    return [errornorm(v_exp, all_u[i]) for i in range(len(all_u) - 1)]


def energy(timescale, timestep, number_of_triangles=30, rheology="VP", solver="FE", stabilised=0):
    """
    Compute the energy of the solution
    u1 - energy defined pg 8, after energy proof
    u2 - energy defined pg 19
    u3 - energy used on the y axis of the energy plot fig 7, pg 20
    """
    all_u, all_h, all_a, mesh, zeta = vp_evp_test_explicit(timescale, timestep, number_of_triangles, rheology,
                                                           advection=True, solver=solver, subcycle=5,
                                                           stabilised=stabilised)
    eta = zeta * e ** (-2)
    energy_u1 = [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u))]
    energy_u2 = [norm(sqrt(zeta) * all_u[i]) for i in range(len(all_u))]
    energy_u3 = [norm(sqrt(eta) * grad(all_u[i])) for i in range(len(all_u))]

    # energy_h = [norm(0.5 * zeta * grad(all_h[i])) for i in range(len(all_h))]
    # energy_a = [norm(0.5 * zeta * grad(all_a[i])) for i in range(len(all_a))]

    return energy_u1, energy_u2, energy_u3


def plot_energies(timescale, timestep):
    "Plotting the energies of the solutions for the energy norm "
    t = np.arange(0, timescale, timestep)
    energy_vp = energy(timescale, timestep, rheology="VP")[0]
    energy_evp = energy(timescale, timestep, rheology="EVP", solver="EVP")[0]
    plt.semilogy(t, energy_vp, label="VP")
    plt.semilogy(t, energy_evp, label="EVP")
    plt.ylabel(r'Energy of solution : $||\frac{\zeta}{2} \nabla v||^2$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Energy of computed solution for Section 4.2 Test, k = {}, T = {}'.format(timestep, timescale))
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_energy.png')


def plot_energies2(timescale, timestep):
    t = np.arange(0, timescale, timestep)
    energy_vp = energy(timescale, timestep, rheology="VP")[1]
    energy_evp = energy(timescale, timestep, rheology="EVP", solver="EVP")[1]
    plt.semilogy(t, energy_vp, label="VP")
    plt.semilogy(t, energy_evp, label="EVP")
    plt.ylabel(r'Energy of solution : $||\sqrt{\zeta} v||^2$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Energy of computed solution for Section 4.2 Test, k = {}, T = {}'.format(timestep, timescale))
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_energy.png')


def plot_energies3(timescale, timestep):
    # paper uses 24 hours as the timescale 86400 [s]
    t = np.arange(0, timescale, timestep)

    # computing the energies
    energy_vp = energy(timescale, timestep, rheology="VP")[2]
    energy_evp = energy(timescale, timestep, rheology="EVP", solver="EVP")[2]
    energy_mevp = energy(timescale, timestep, rheology="VP", solver="mEVP")[2]
    energy_vp_stab = energy(timescale, timestep, rheology="VP", stabilised=1)[2]
    energy_evp_stab = energy(timescale, timestep, rheology="EVP", solver="EVP", stabilised=1)[2]
    energy_mevp_stab = energy(timescale, timestep, rheology="VP", solver="mEVP", stabilised=1)[2]

    plt.semilogy(t, energy_vp, label="VP")
    plt.semilogy(t, energy_evp, label="EVP")
    plt.semilogy(t, energy_mevp, label="mEVP")
    plt.semilogy(t, energy_vp_stab, label="VP stabilised")
    plt.semilogy(t, energy_evp_stab, label="EVP stabilised")
    plt.semilogy(t, energy_mevp_stab, label="mEVP stabilised")

    plt.ylabel(r'Energy of solution: $||\sqrt{\eta} \nabla v||^2$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Energy of computed solution for Section 4.2 Test, k = {}, T = {}'.format(timestep, timescale))
    plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_energy.png')


start = time.time()
plot_energies3(10, 1)
# plot_energies3(86400, 100)
end = time.time()
print(end - start)