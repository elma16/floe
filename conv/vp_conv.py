import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.vp_evp_rheology import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

def vp_error(timescale,timestep,number_of_triangles = 30,stabilised = 0):
    """
    Compute the error norm of the velocity against the stationary solution for test 1
    """
    all_u,mesh,v_exp,zeta = vp_evp_test1(timescale,timestep,number_of_triangles,stabilised)
    return [errornorm(v_exp, all_u[i]) for i in range(len(all_u)-1)]

def energy(timescale,timestep,number_of_triangles=30):
    all_u, all_h, all_a, mesh, zeta = vp_evp_test1(timescale, timestep, number_of_triangles,advection=True)
    energy_u = [norm(0.5 * zeta * grad(all_u[i])) for i in range(len(all_u) )]
    energy_h = [norm(0.5 * zeta * grad(all_h[i])) for i in range(len(all_h) )]
    energy_a = [norm(0.5 * zeta * grad(all_a[i])) for i in range(len(all_a) )]

    return energy_u,energy_h,energy_a

def plot_energies(timescale,timestep):
    t = np.arange(0,timescale,timestep)
    for j in range(3):
        plt.semilogy(t, energy(timescale, timestep)[j], label = "variable = %s" % j)
        plt.ylabel(r'Energy of solution ')
        plt.xlabel(r'Time [s]')
        plt.title(r'Energy of computed solution for Section 4.2 Test, k = {}, T = {}'.format(timestep,timescale))
        plt.legend(loc='best')
    plt.show()
    plt.savefig('./plots/strain_rate_energy.png')

plot_energies(100,1)
