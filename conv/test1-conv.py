import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.strain_rate_tensor import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

'''
Convergence plots for the strain rate tensor test:
    u vs. t (timestep variable)
    u vs. t (meshsize variable)
    u vs. t (stabilised vs. unstabilised)
'''


def strain_rate_tensor_error(timescale, timestep, number_of_triangles=35, stabilised=0):
    """
    Compute the error norm of the velocity against the stationary solution for test 1
    """
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    return [errornorm(v_exp, all_u[i]) for i in range(len(all_u) - 1)]


def plot_u_conv_vs_timestep():
    # plotting the convergence of velocity with timestep fixed, and timescale changing
    for k in [10, 100, 1000]:
        t = np.arange(0, k, 10)
        plt.semilogy(t, strain_rate_tensor_error(k, timestep=10), label="timescale = %s" % k)
        plt.ylabel(r'Error of solution ')
        plt.xlabel(r'Time [s]')
        plt.title(r'Error of computed solution for Section 4.1 Test, k = {}, T = {}'.format(k, k))
        plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_velo_t={y}.png'.format(y=k))


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


def energy(timescale, timestep, number_of_triangles=35):
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles)
    return [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u) - 1)]


def plot_u_energy(timescale, timestep):
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
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    return [gt(norm(grad(all_u[i])),sqrt(norm(grad(all_u[i])+transpose(grad(all_u[i]))))) for i in range(len(all_u))]

def vel_comp_max(timescale, timestep, number_of_triangles=35, stabilised=0):
    """
    Computing the maximum component of all the velocities in the velocity field
    """
    all_u, mesh, v_exp, zeta = strain_rate_tensor(timescale, timestep, number_of_triangles, stabilised)
    print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])

vel_comp_max(10,1)