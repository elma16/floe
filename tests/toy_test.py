from firedrake import *

def toy_problem(T=10,timestep=10**(-3),stabilised=0,number_of_triangles=30,output=False,shape = "Square"):
    """
    A trial toy test problem where I start off with a big square in the middle of the velocity field
    to demonstrate the nature of hyperbolic PDEs
    """
    print('\n******************************** TOY PROBLEM ********************************\n')

    L = 500000
    mesh = SquareMesh(number_of_triangles, number_of_triangles, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    if shape == "Half-Plane":
        #half plane
        u_.interpolate(conditional(le(x,L/2),as_vector([10,10]),as_vector([0,0])))
    elif shape == "Square":
        # square
        u_.interpolate(conditional(le(abs(x-L/2)+abs(y-L/2),L/5), as_vector([10, 10]), as_vector([0, 0])))
    elif shape == "Circle":
        # circle
        u_.interpolate(conditional(le(((x-L/2)**2+(y-L/2)**2),10000*L), as_vector([10, 10]), as_vector([0, 0])))


    u.assign(u_)

    h = Constant(1)

    A = Constant(1)

    # defining the constants to be used in the sea ice momentum equation:

    # ice strength parameter
    P_star = Constant(27.5 * 10 ** 3)

    # ice concentration parameter
    C = Constant(20)

    # ice strength
    P = P_star * h * exp(-C * (1 - A))

    Delta_min = Constant(2 * 10 ** (-9))

    # viscosities
    zeta = P / (2 * Delta_min)

    # internal stress tensor, stabilised vs unstabilised
    if stabilised == 0:
        sigma = zeta / 2 * (grad(u) + transpose(grad(u)))
    elif stabilised == 1:
        sigma = avg(CellVolume(mesh))/FacetArea(mesh)*(dot(jump(u),jump(v)))*dS
    elif stabilised == 2:
        sigma = zeta / 2 * (grad(u))
    else:
        raise ValueError("Expected 0, 1 or 2 but got {:d}".format(stabilised))


    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    #step used in the rate of strain tensor
    #u_.interpolate(v_exp)
    #u.assign(u_)

    sigma_exp = zeta / 2 * (grad(v_exp) + transpose(grad(v_exp)))
    R = -div(sigma_exp)

    def strain(omega):
        return 1 / 2 * (omega + transpose(omega))

    # momentum equation
    a = (inner((u - u_) / timestep, v) + inner(sigma, strain(grad(v)))) * dx
    a -= inner(R, v) * dx

    t = 0.0

    all_errors = []
    end = T
    bcs = [DirichletBC(V, 0, "on_boundary")]
    params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu"}

    if output:
        outfile = File('./output/toy/toy_u_circle.pvd')
        outfile.write(u_, time=t)
        print('******************************** Forward solver ********************************\n')
        while t <= end:
            solve(a == 0, u,solver_parameters = params,bcs = bcs)
            u_.assign(u)
            t += timestep
            error = norm(u - v_exp)
            outfile.write(u_, time=t)
            print("Time:", t, "[s]")
            print(int(min(t / T * 100,100)), "% complete")
            print("Error norm:", error)
            all_errors.append(error)
        print('... forward problem solved...\n')
    else:
        print('******************************** Forward solver ********************************\n')
        while t <= end:
            solve(a == 0, u, solver_parameters=params, bcs=bcs)
            u_.assign(u)
            t += timestep
            error = norm(u - v_exp)
            print("Time:", t, "[s]")
            print(int(min(t / T * 100, 100)), "% complete")
            print("Error norm:", error)
            all_errors.append(error)
        print('... forward problem solved...\n')


    del all_errors[-1]
    print('...done!')
    return all_errors

toy_problem(T=1,timestep=10**(-2),output=True)