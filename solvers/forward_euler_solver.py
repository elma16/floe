from firedrake import *

from solvers.solver_parameters import *

def forward_euler_solver(u,u_,lm,bcs,t,timestep,timescale,pathname,output=False,advection=False,stabilisation = False,lh=None,la=None,h=None,h_=None,a=None,a_=None):
    if not advection:
        if output:
            outfile = File('{pathname}'.format(pathname = pathname))
            outfile.write(u_, time=t)
            print('******************************** Forward solver ********************************\n')
            while t <= timescale:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                t += timestep
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t <= timescale:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
    if advection:
        if output:
            outfile = File('{pathname}'.format(pathname = pathname))
            outfile.write(u_, time=t)
            print('******************************** Forward solver ********************************\n')
            while t <= timescale:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                solve(lh == 0, h, solver_parameters=params, bcs=bcs)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params, bcs=bcs)
                a_.assign(a)
                t += timestep
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t <= timescale:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                solve(lh == 0, h, solver_parameters=params, bcs=bcs)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params, bcs=bcs)
                a_.assign(a)
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')

def forward_euler_solver_error(u,u_,v_exp,lm,bcs,t,timestep,timescale,pathname,output=False):
    all_errors = []

    if output:
        outfile = File('{pathname}'.format(pathname = pathname))
        outfile.write(u_, time=t)
        print('******************************** Forward solver ********************************\n')
        while t <= timescale:
            solve(lm == 0, u, solver_parameters = params, bcs = bcs)
            u_.assign(u)
            t += timestep
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            print("Error norm:", errornorm(v_exp,u))
            all_errors.append(errornorm(v_exp,u))
        print('... forward problem solved...\n')
    else:
        print('******************************** Forward solver (NO OUTPUT) ********************************\n')
        while t <= timescale:
            solve(lm == 0, u, solver_parameters=params, bcs=bcs)
            u_.assign(u)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            print("Error norm:", errornorm(v_exp,u))
            all_errors.append(errornorm(v_exp,u))
        print('... forward problem solved...\n')

    return all_errors