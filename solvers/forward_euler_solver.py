from firedrake import *

from solvers.solver_parameters import *

def forward_euler_solver(u,u_,lm,bcs,t,timestep,timescale,pathname,output=False,advection=False,lh=None,la=None,h=None,h_=None,a=None,a_=None):
    all_u = []
    all_h = []
    all_a = []
    if not advection:
        if output:
            outfile = File('{pathname}'.format(pathname = pathname))
            outfile.write(u_, time=t)
            print('******************************** Forward solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                all_u.append(Function(u))
                t += timestep
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                all_u.append(Function(u))
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
    if advection:
        if output:
            outfile = File('{pathname}'.format(pathname = pathname))
            outfile.write(u_, time=t)
            print('******************************** Forward solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                solve(lh == 0, h, solver_parameters=params)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params)
                a_.assign(a)
                all_u.append(Function(u))
                all_h.append(Function(h))
                all_a.append(Function(a))
                t += timestep
                outfile.write(u_, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u, solver_parameters=params, bcs=bcs)
                u_.assign(u)
                solve(lh == 0, h, solver_parameters=params)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params)
                a_.assign(a)
                all_u.append(Function(u))
                all_h.append(Function(h))
                all_a.append(Function(a))
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')

    return all_u,all_h,all_a