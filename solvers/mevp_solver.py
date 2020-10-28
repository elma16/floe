from firedrake import *

from solvers.solver_parameters import *

def mevp_stress_solver(sigma,ep_dot,zeta,P):
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
    sigma1 = 1 + (sigma1 + 2 * zeta * (ep_dot1 - P)) / alpha
    sigma2 = 1 + (sigma2 * zeta * ep_dot2) / 2 * alpha

    sigma = as_matrix([[0.5 * (sigma1 + sigma2),1 + (sigma[0, 1] * zeta * ep_dot[0, 1]) / 2 * alpha],
                       [1 + (sigma[0, 1] * zeta * ep_dot[0, 1]) / 2 * alpha,0.5 * (sigma1 - sigma2) ]])

    return sigma

def mevp_solver(u,u_,a,t,timestep,subcycle,bcs,sigma,ep_dot,P,zeta,T,timescale,pathname,output=False):
    subcycle_timestep = timestep / subcycle
    if output:
        outfile = File('{pathname}'.format(pathname = pathname))
        outfile.write(u_, time=t)

        print('******************************** mEVP Solver ********************************\n')
        while t <= timescale:
            s = t
            while s <= t + timestep:
                solve(a == 0, u, solver_parameters=params, bcs=bcs)
                mevp_stress_solver(sigma, ep_dot, P, zeta)
                u_.assign(u)
                s += subcycle_timestep
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... mEVP problem solved...\n')
    else:
        print('******************************** mEVP Solver (NO OUTPUT) ********************************\n')
        while t <= timescale:
            s = t
            while s <= t + timestep:
                solve(a == 0, u, solver_parameters=params, bcs=bcs)
                mevp_stress_solver(sigma, ep_dot, P, zeta)
                u_.assign(u)
                s += subcycle_timestep
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... mEVP problem solved...\n')