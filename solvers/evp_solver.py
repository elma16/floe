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
    sigma1 = (2 * subcycle_timestep * zeta * ep_dot1 - P / 2 + 2 * T * sigma1) / (2 * T + subcycle_timestep)
    sigma2 = (2 * subcycle_timestep * zeta * ep_dot2 - P / 2 + 2 * T * sigma2) / (2 * T + subcycle_timestep)

    # compose the new sigma in terms of the old sigma components
    sigma = as_matrix([[0.5 * (sigma1 + sigma2),
                        (subcycle_timestep * zeta * ep_dot[0, 1] + T * sigma[0, 1]) / (T + 2 * subcycle_timestep)],
                       [(subcycle_timestep * zeta * ep_dot[0, 1] + T * sigma[0, 1]) / (T + 2 * subcycle_timestep),
                        0.5 * (sigma1 - sigma2)]])

    return sigma


def evp_solver(u, u_, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, T, timescale, pathname, output=False,
               advection=False, lh=None, la=None, h=None, h_=None, a=None, a_=None):
    subcycle_timestep = timestep / subcycle
    all_u = []
    all_h = []
    all_a = []
    if not advection:
        if output:
            outfile = File('{pathname}'.format(pathname=pathname))
            outfile.write(u_, time=t)

            print('******************************** EVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                    evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u_.assign(u)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u))
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
        else:
            print('******************************** EVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                    evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u_.assign(u)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
    if advection:
        if output:
            outfile = File('{pathname}'.format(pathname=pathname))
            outfile.write(u_, time=t)

            print('******************************** EVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                    evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u_.assign(u)
                    solve(lh == 0, h, solver_parameters=params)
                    h_.assign(h)
                    solve(la == 0, a, solver_parameters=params)
                    a_.assign(a)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u))
                all_h.append(Function(h))
                all_a.append(Function(a))
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
        else:
            print('******************************** EVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                    evp_stress_solver(sigma, ep_dot, P, zeta, T, subcycle_timestep=s)
                    u_.assign(u)
                    solve(lh == 0, h, solver_parameters=params)
                    h_.assign(h)
                    solve(la == 0, a, solver_parameters=params)
                    a_.assign(a)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u))
                all_h.append(Function(h))
                all_a.append(Function(a))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... EVP problem solved...\n')
    return all_u, all_h, all_a
