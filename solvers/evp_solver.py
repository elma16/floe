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


def evp_solver(u1, u0, usolver, t, timestep, subcycle, sigma, ep_dot, P, zeta, T, timescale, output=False,
               advection=False, hsolver=None, asolver=None, h1=None, h0=None, a1=None, a0=None):
    subcycle_timestep = timestep / subcycle
    all_u = []
    all_h = []
    all_a = []
    pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)
    if not advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)

            print('******************************** EVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    usolver.solve()
                    sigma = evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
        else:
            print('******************************** EVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    usolver.solve()
                    sigma = evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
    if advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)

            print('******************************** EVP Solver ********************************\n')
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
                all_u.append(Function(u1))
                all_h.append(Function(h1))
                all_a.append(Function(a1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
        else:
            print('******************************** EVP Solver (NO OUTPUT) ********************************\n')
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
                all_u.append(Function(u1))
                all_h.append(Function(h1))
                all_a.append(Function(a1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
    return all_u, all_h, all_a
