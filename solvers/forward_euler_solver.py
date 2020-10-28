from firedrake import *

from solvers.solver_parameters import *

def forward_euler_solver(u,u_,a,bcs,t,timestep,timescale,pathname,output="False"):
    if output:
        outfile = File('{pathname}'.format(pathname = pathname))
        outfile.write(u_, time=t)
        print('******************************** Forward solver ********************************\n')
        while t <= timescale:
            solve(a == 0, u, solver_parameters=params, bcs=bcs)
            u_.assign(u)
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        print('... forward problem solved...\n')
    else:
        print('******************************** Forward solver (NO OUTPUT) ********************************\n')
        while t <= timescale:
            solve(a == 0, u, solver_parameters=params, bcs=bcs)
            u_.assign(u)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        print('... forward problem solved...\n')
