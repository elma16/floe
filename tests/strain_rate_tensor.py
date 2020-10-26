from firedrake import *
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

def strain_rate_tensor(length_of_time=10,timestep=10**(-6),stabilised=0,number_of_triangles=100):
    '''
    from Mehlmann and Korn, 2020
    Section 4.2
    L = 500000
    pi_x = pi_y = pi/L
    By construction, the analytical solution is
        v_1 = -sin(pi_x*x)*sin(pi_y*y)
        v_2 = -sin(pi_x*x)*sin(pi_y*y)
    zeta = P/2*Delta_min

    number_of_triangles: paper's value for 3833 edges is between 35,36.

    stabilised = {0,1,2}
    0 - unstabilised (default option)
    1 - stabilised (change the form of the stress tensor)
    2 - stabilised (via the a velocity jump algorithm)
    '''

    n = number_of_triangles
    L = 500000
    mesh = SquareMesh(n, n, L)

    V = VectorFunctionSpace(mesh, "CR", 1)

    # sea ice velocity
    u_ = Function(V, name="Velocity")
    u = Function(V, name="VelocityNext")

    # test functions
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # initial conditions

    #u_.assign(as_vector([0, 0]))

    #u.assign(u_)

    h = Constant(1)

    A = Constant(1)

    timestep = timestep

    T = length_of_time

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
        #algo
    elif stabilised == 2:
        sigma = zeta / 2 * (grad(u))
    else:
        return("Not a valid input. Try again.")


    pi_x = pi / L

    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    u_.interpolate(v_exp)
    u.assign(u_)

    sigma_exp = zeta / 2 * (grad(v_exp) + transpose(grad(v_exp)))
    R = -div(sigma_exp)

    def strain(omega):
        return 1 / 2 * (omega + transpose(omega))

    # momentum equation
    a = (inner((u - u_) / timestep, v) + inner(sigma, strain(grad(v)))) * dx
    a -= inner(R, v) * dx

    t = 0.0

    ufile = File('strain_rate_tensor_u_no_norm.pvd')
    ufile.write(u_, time=t)
    all_errors = []
    end = T
    bcs = [DirichletBC(V, 0, "on_boundary")]

    while (t <= end):
        solve(a == 0, u,
                solver_parameters={"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu"},
                bcs=bcs)
        u_.assign(u)
        t += timestep
        error = norm(u - v_exp)
        ufile.write(u_, time=t)
        print("Time:", t, "seconds", min(t / T * 100,100), "% complete")
        print("Norm:", error)
        all_errors.append(error)

    del all_errors[-1]

    return all_errors




'''
Creating all the vector plots and plotting error against time.
'''


starttime = time.time()
all_errors1 = strain_rate_tensor(10**(-2),10**(-4),error_plot = True)
all_errors2 = strain_rate_tensor(10**(-2),10**(-4),error_plot = True,number_of_triangles=10)
all_errors3 = strain_rate_tensor(10**(-2),10**(-4),error_plot = True,stabilised=True)
all_errors4 = strain_rate_tensor(10**(-2),10**(-4),error_plot = True,stabilised=True,number_of_triangles=10)
endtime = time.time()
print(endtime-starttime)

length_of_time = 10**(-2)
timestep = 10**(-4)
t = np.arange(0, length_of_time, timestep)
plt.plot(t, all_errors1,'r--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
plt.plot(t,all_errors2,'b.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
plt.plot(t,all_errors3,'g--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v)$')
plt.plot(t,all_errors4,'k.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v)$')
plt.ylabel(r'Error of solution $[\times 10^3]$')
plt.xlabel(r'Time [s]')
plt.title(r'Error of computed solution for Section 4.1 Test, $k = 10^{-4}, T = 10^{-2}$')
plt.legend(loc='best')
plt.show()
