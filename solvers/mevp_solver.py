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


def mevp_solver(u1, u0, usolver, t, timestep, subcycle, sigma, ep_dot, P, zeta, timescale,
                advection=False, hsolver=None, asolver=None, h1=None, h0=None, a1=None, a0=None):
    subcycle_timestep = timestep / subcycle
    ndump = 10
    dumpn = 0
    pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)
    outfile = File(pathname)
    outfile.write(u1, time=t)
    print('******************************** mEVP Solver ********************************\n')
    if not advection:
        while t < timescale - 0.5 * timestep:
            s = t
            while s <= t + timestep:
                usolver.solve()
                sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
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
                sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
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

    print('... mEVP problem solved...\n')
