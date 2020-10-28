from firedrake import *

def EVPsolver(sigma,ep_dot,P,zeta,T,subcycle_timestep):
    """
    Implementation of the EVP solver used by Mehlmann and Korn:

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
    sigma = as_matrix([[0.5*(sigma1 + sigma2),( subcycle_timestep * zeta * ep_dot[0,1] + T * sigma[0,1]) / (T + 2 * subcycle_timestep)],
                       [( subcycle_timestep * zeta * ep_dot[0,1] + T * sigma[0,1]) / (T + 2 * subcycle_timestep),0.5*(sigma1 - sigma2)]])

    return sigma