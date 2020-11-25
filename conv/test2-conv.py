import os, sys, inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tests.vp_evp_rheology import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")


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
