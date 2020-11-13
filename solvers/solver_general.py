from firedrake import *

from solvers.solver_parameters import *

t = 0.0
all_u = []

def ievp_solver(output,last_frame,pathname,timescale,timestep,t,w0,w1,u1,usolver):
    if output and last_frame:
        ufile = File(pathname)
        ufile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            all_u.append(Function(u1))

        print('... EVP problem solved...\n')
        ufile.write(u1, time=t)

    elif output and not last_frame:
        ufile = File(pathname)
        ufile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
            ufile.write(u1, time=t)
            all_u.append(Function(u1))
        print('... EVP problem solved...\n')
    else:
        while t < timescale - 0.5 * timestep:
            t += timestep
            usolver.solve()
            w0.assign(w1)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        print('... EVP problem solved...\n')
    print('...done!')
    return all_u

def imevp(output,last_frame,timescale,timestep,u0,t,usolver,ssolver,u1,pathname):
    if output and last_frame:
        outfile = File(pathname)
        outfile.write(u0, time=t)

        print('******************************** Implicit EVP Solver ********************************\n')
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            ssolver.solve()
            u0.assign(u1)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
        outfile.write(u0, time=t)
        print('... EVP problem solved...\n')
    elif output and not last_frame:
        outfile = File(pathname)
        outfile.write(u0, time=t)

        print('******************************** Implicit EVP Solver ********************************\n')
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            ssolver.solve()
            u0.assign(u1)
            t += timestep
            outfile.write(u0, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')
    else:
        print('******************************** Implicit EVP Solver (NO OUTPUT) ********************************\n')
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            ssolver.solve()
            u0.assign(u1)
            t += timestep
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

        print('... EVP problem solved...\n')