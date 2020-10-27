def mEVPsolver(sigma,ep_dot,zeta,P,T,subcycles,subcycle_timestep):
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
    sigma[0, 1] = 1 + (sigma[0, 1] * zeta * ep_dot[0, 1]) / 2 * alpha

    # computing the entries of the stress tensor
    sigma[1, 0] = sigma[0, 1]
    sigma[0, 0] = (sigma1 + sigma2) / 2
    sigma[1, 1] = (sigma1 - sigma2) / 2

    return sigma