from firedrake import *


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


def evp_solver(u1, u0, usolver, t, timestep, subcycle, sigma, ep_dot, P, zeta, T, timescale,
               advection=False, hsolver=None, asolver=None, h1=None, h0=None, a1=None, a0=None, modified=False):
    subcycle_timestep = timestep / subcycle
    ndump = 10
    dumpn = 0
    pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)
    outfile = File(pathname)

    print('******************************** EVP Solver ********************************\n')
    if advection:
        outfile.write(u1, h1, a1, time=t)
        while t < timescale - 0.5 * timestep:
            s = t
            while s <= t + timestep:
                usolver.solve()
                if modified:
                    sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
                else:
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
                outfile.write(u1, h1, a1, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
    else:
        outfile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            s = t
            while s <= t + timestep:
                usolver.solve()
                if modified:
                    sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
                else:
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

    print('... EVP problem solved...\n')
