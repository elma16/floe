from firedrake import *

from solvers.solver_parameters import *


def forward_euler_solver(u1, u0, lm, bcs, t, timestep, timescale, pathname, output=False, advection=False, lh=None,
                         la=None, h=None, h_=None, a=None, a_=None):
    all_u = []
    all_h = []
    all_a = []
    if not advection:
        if output:
            outfile = File('{pathname}'.format(pathname=pathname))
            outfile.write(u0, time=t)
            print('******************************** Forward solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                u0.assign(u1)
                all_u.append(Function(u1))
                t += timestep
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                u0.assign(u1)
                all_u.append(Function(u1))
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
    if advection:
        if output:
            outfile = File('{pathname}'.format(pathname=pathname))
            outfile.write(u0, time=t)
            print('******************************** Forward solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                u0.assign(u1)
                solve(lh == 0, h, solver_parameters=params)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params)
                a_.assign(a)
                all_u.append(Function(u1))
                all_h.append(Function(h))
                all_a.append(Function(a))
                t += timestep
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')
        else:
            print('******************************** Forward solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                solve(lm == 0, u1, solver_parameters=params, bcs=bcs)
                u0.assign(u1)
                solve(lh == 0, h, solver_parameters=params)
                h_.assign(h)
                solve(la == 0, a, solver_parameters=params)
                a_.assign(a)
                all_u.append(Function(u1))
                all_h.append(Function(h))
                all_a.append(Function(a))
                t += timestep
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")
            print('... forward problem solved...\n')

    return all_u, all_h, all_a
