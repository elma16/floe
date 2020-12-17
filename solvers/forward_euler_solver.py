from firedrake import *


def forward_euler_solver(u1, u0, usolver, t, timestep, timescale, advection=False, hsolver=None, asolver=None,
                         h1=None, h0=None, a1=None, a0=None):
    all_u = []
    all_h = []
    all_a = []
    ndump = 10
    dumpn = 0

    pathname = "./output/strain_rate_tensor/test_T={}_k={}_N={}_stab={}.pvd"
    #pathname = './output/vp_evp_test/{}test_{}.pvd'.format(timescale, timestep)

    outfile = File(pathname)

    print('******************************** Forward solver ********************************\n')
    if advection:
        outfile.write(u1, h1, a1, time=t)
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            u0.assign(u1)
            hsolver.solve()
            h0.assign(h1)
            asolver.solve()
            a0.assign(a1)
            all_u.append(Function(u1))
            all_h.append(Function(h1))
            all_a.append(Function(a1))
            t += timestep
            dumpn += 1
            if dumpn == ndump:
                dumpn -= ndump
                outfile.write(u1, h1, a1, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")
    else:
        outfile.write(u1, time=t)
        while t < timescale - 0.5 * timestep:
            usolver.solve()
            u0.assign(u1)
            all_u.append(Function(u1))
            t += timestep
            dumpn += 1
            if dumpn == ndump:
                dumpn -= ndump
                outfile.write(u1, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / timescale * 100, 100)), "% complete")

    print('... forward problem solved...\n')
    return all_u, all_h, all_a
