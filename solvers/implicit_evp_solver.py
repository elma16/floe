from firedrake import *

from solvers.solver_parameters import *

def evp_solve2(timestep,e,zeta,T,ep_dot,sigma,P):
    sigma00 = timestep*zeta/T*ep_dot[0,0]-timestep*P/4*T+(1-timestep*e**2/4*T)*sigma[0,0]-(timestep*(1-e**2)/8*T)*tr(sigma)
    sigma11 = timestep*zeta/T*ep_dot[1,1]-timestep*P/4*T+(1-timestep*e**2/4*T)*sigma[1,1]-(timestep*(1-e**2)/8*T)*tr(sigma)
    sigma01 = timestep*zeta/T*ep_dot[0,1]-timestep*P/4*T+(1-timestep*e**2/4*T)*sigma[0,1]
    sigma = as_matrix([[sigma00,sigma01],[sigma01,sigma11]])
    return sigma