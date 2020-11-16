import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from firedrake import *

from solvers.mevp_solver import *

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

def bt_solver(output,u0,u1,t,t0,timestep,timescale,usolver,sigma,ep_dot,P,zeta,subcycle,advection=False,hsolver=None,h0=None,h1=None,asolver=None,a0=None,a1=None):

    pathname = "./output/box_test/box_test_exp.pvd"

    subcycle_timestep = timestep / subcycle

    if not advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)
            print('******************************** mEVP Solver ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    usolver.solve()
                    sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                t0.assign(t)
                all_u.append(Function(u1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
        else:
            print('******************************** mEVP Solver (NO OUTPUT) ********************************\n')
            while t < timescale - 0.5 * timestep:
                s = t
                while s <= t + timestep:
                    usolver.solve()
                    sigma = mevp_stress_solver(sigma, ep_dot, P, zeta)
                    u0.assign(u1)
                    s += subcycle_timestep
                t += timestep
                t0.assign(t)
                all_u.append(Function(u1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
    if advection:
        if output:
            outfile = File(pathname)
            outfile.write(u0, time=t)

            print('******************************** mEVP Solver ********************************\n')
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
                t0.assign(t)
                all_u.append(Function(u1))
                outfile.write(u0, time=t)
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')
        else:
            print('******************************** mEVP Solver (NO OUTPUT) ********************************\n')
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
                t0.assign(t)
                all_u.append(Function(u1))
                print("Time:", t, "[s]")
                print(int(min(t / timescale * 100, 100)), "% complete")

            print('... mEVP problem solved...\n')

def taylor_galerkin(timescale,timestep,t):
    """
    The Taylor-Galerkin method for the FE discretisation of ice transport equations:
    Given u^n,a^n, compute a^{n+1}.
    """
    L = 500000
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)
    U = FunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u0 = Function(V)
    u1 = Function(V)

    # mean height of sea ice
    h0 = Function(U)
    h1 = Function(U)

    # sea ice concentration
    a0 = Function(U)
    a1 = Function(U)

    # test functions
    v = TestFunction(V)
    w = TestFunction(U)

    x, y = SpatialCoordinate(mesh)

    gn = u0*a0-0.5*timestep*u0*grad(u0*a0)

    la = inner(a1 - a0 + timestep*grad(gn),w)*dx

    aprob = NonlinearVariationalProblem(la,a1)
    asolver = NonlinearVariationalSolver(aprob,solver_parameters = params)
    while t < timescale - 0.5 * timestep:
        return None

