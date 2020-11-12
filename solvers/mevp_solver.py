from firedrake import *

from solvers.solver_parameters import *


def mevp_stress_solver(sigma, ep_dot, zeta, P):
    """
    Implementation of the mEVP solver used by Mehlmann and Korn:

    Don't forget that the zeta term depends on v, and so changes in time!
    """

    # defining the terms used in the mEVP solver
    sigma1 = sigma[0, 0] + sigma[1, 1]
    sigma2 = sigma[0, 0] - sigma[1, 1]
    ep_dot1 = ep_dot[0, 0] + ep_dot[1, 1]
    ep_dot2 = ep_dot[0, 0] - ep_dot[1, 1]
    alpha = Constant(500)

    # updating the mEVP stress tensor
    sigma1 = ((alpha + 1) * sigma1 + 2 * zeta * (ep_dot1 - P)) / alpha
    sigma2 = sigma2 * (1 + (zeta * ep_dot2) / (2 * alpha))

    sigma = as_matrix([[0.5 * (sigma1 + sigma2),
                        0.5 * sigma2 * (1 + (zeta * ep_dot2) / alpha)],
                       [0.5 * sigma2 * (1 + (zeta * ep_dot2)) / alpha, 0.5 * (sigma1 - sigma2)]])

    return sigma


def mevp_solver(u1, u0, lm, t, timestep, subcycle, bcs, sigma, ep_dot, P, zeta, timescale, output=False,
                advection=False, lh=None, la=None, h1=None, h0=None, a1=None, a0=None):
    subcycle_timestep = timestep / subcycle
    all_u = []
    all_h = []
    all_a = []
    pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)
    if not advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)

            print('******************************** mEVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                    mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
        else:
            print('******************************** mEVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                    mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
    if advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)

            print('******************************** mEVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                    mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    solve(lh == 0, h1, solver_parameters=params)
                    h0.assign(h1)
                    solve(la == 0, a1, solver_parameters=params)
                    a0.assign(a1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                all_h.append(Function(h1))
                all_a.append(Function(a1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
        else:
            print('******************************** mEVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                    mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    solve(lh == 0, h1, solver_parameters=params)
                    h0.assign(h1)
                    solve(la == 0, a1, solver_parameters=params)
                    a0.assign(a1)
                    s += subcycle_timestep
                t += timestep
                all_u.append(Function(u1))
                all_h.append(Function(h1))
                all_a.append(Function(a1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')

    return all_u, all_h, all_a
