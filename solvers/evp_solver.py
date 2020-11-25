from firedrake import *

from solvers.solver_parameters import *


def evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep):
    """
    Implementation of the stress tensor solver used by Mehlmann and Korn:

    """

    # defining the terms used in the EVP solver
    sigma1 = sigma[0, 0] + sigma[1, 1]
    sigma2 = sigma[0, 0] - sigma[1, 1]
    ep_dot1 = ep_dot[0, 0] + ep_dot[1, 1]
    ep_dot2 = ep_dot[0, 0] - ep_dot[1, 1]

    # solve for the next subcycle timestep
    sigma1 = (subcycle_timestep * (2 * zeta * ep_dot1 - P) + 2 * T * sigma1) / (2 * T + subcycle_timestep)
    sigma2 = (subcycle_timestep * zeta * ep_dot2 + T * sigma2) / (T + 2 * subcycle_timestep)

    # compose the new sigma in terms of the old sigma components
    sigma = as_matrix([[0.5 * (sigma1 + sigma2),
                        (subcycle_timestep * zeta * ep_dot[0, 1] + T * sigma[0, 1]) / (T + 2 * subcycle_timestep)],
                       [(subcycle_timestep * zeta * ep_dot[0, 1] + T * sigma[0, 1]) / (T + 2 * subcycle_timestep),
                        0.5 * (sigma1 - sigma2)]])

    return sigma


def evp_solver(u1, u0, usolver, t, timestep, subcycle, sigma, ep_dot, P, zeta, T, timescale,
               advection=False, hsolver=None, asolver=None, h1=None, h0=None, a1=None, a0=None):
    subcycle_timestep = timestep / subcycle
    pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)
    ndump = 10
    dumpn = 0
    outfile = File(pathname)
    outfile.write(u1, time=t)

    print('******************************** EVP Solver ********************************\n')
    if not advection:
        while t < timescale - 0.5 * timestep:
            s = t
            while s <= t + timestep:
                usolver.solve()
                sigma = evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                u0.assign(u1)
                s += subcycle_timestep
            t += timestep
            dumpn += 1
            if dumpn == ndump:
                dumpn -= ndump
                outfile.write(u1, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
    if advection:
        while t < timescale - 0.5 * timestep:
            s = t
            while s <= t + timestep:
                usolver.solve()
                sigma = evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                u0.assign(u1)
                hsolver.solve()
                h0.assign(h1)
                asolver.solve()
                a0.assign(a1)
                s += subcycle_timestep
            t += timestep
            dumpn += 1
            if dumpn == ndump:
                dumpn -= ndump
                outfile.write(u1, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
    print('... EVP problem solved...\n')
