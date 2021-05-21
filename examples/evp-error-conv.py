from seaice import *
from firedrake import *
from pathlib import Path
import matplotlib.pyplot as plt

path = "./output/evp-error-conv"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP Error Convergence

Manufactured solutions.
Coriolis force neglected, no forcing due to wind. 
Forcing due to ocean is present.
Advection turned off.
Boundary conditions : u = 0
Initial conditions : h = 1, A = x / L
Domain is a 500km x 500km square.
'''

timestep = 1
dumpfreq =  1
timescale = 10

zero = Constant(0)

title = "EVP Plot"
diagnostic_dirname = path + "/evp.nc"
plot_dirname = path + "/evp_error_timescale={}_timestep={}.png".format(timescale, timestep)
dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

length = 5 * 10 ** 5
pi_x = pi / length

#number_of_triangles = [5, 10, 20, 40, 80]
number_of_triangles = [40]

error_values = []

stabilised = {'state':False, 'alpha':1}
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters(rho_a=zero, C_a=zero, cor=zero)

for values in number_of_triangles:

    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])
    sigma_exp = as_matrix([[-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
                           [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)]])
    #v_exp = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
    #sigma_exp = as_matrix([[1, 1],
    #                       [1, 1]])


    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

    ic =  {'u': v_exp, 'a': 1, 'h': 1, 's': sigma_exp}

    conditions = Conditions(ic=ic, ocean_curr=ocean_curr, stabilised=stabilised, family='CR')

    evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                                solver_params=solver)
    
    #eqn = evp.momentum_equation(evp.h, evp.u1, evp.u0, evp.p, evp.sh, params.rho, evp.uh, conditions.ocean_curr, params.rho_a,
    #                            params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, evp.timestep, ind=evp.ind)

    
    eqn = inner(params.rho * evp.h * (evp.u1 - evp.u0), evp.p) * dx
    eqn += timestep * inner(evp.sh, grad(evp.p)) * dx
    eqn -= timestep * inner(params.rho_w * params.C_w * sqrt(dot(ocean_curr - evp.uh, ocean_curr - evp.uh)) * (ocean_curr - evp.uh), evp.p) * dx

    # source terms due to sigma_exp and v_exp
    eqn += timestep * inner(sigma_exp, grad(evp.p)) * dx
    eqn -= timestep * inner(params.rho_w * params.C_w * sqrt(dot(ocean_curr - v_exp, ocean_curr - v_exp)) * (ocean_curr - v_exp), evp.p) * dx

    zeta_exp = evp.zeta(evp.h, evp.a, evp.delta(v_exp))
    ep_dot_exp = evp.strain(grad(v_exp))
    rheology_exp = params.e ** 2 * sigma_exp + Identity(2) * 0.5 * ((1 - params.e ** 2) * tr(sigma_exp) + evp.Ice_Strength(evp.h, evp.a))
    zeta = evp.zeta(evp.h, evp.a, evp.delta(evp.uh))
    
    eqn += inner(evp.s1 - evp.s0 + 0.5 * timestep * evp.rheology / params.T, evp.q) * dx
    eqn -= inner(evp.q * zeta * timestep / params.T, evp.ep_dot) * dx

    # source terms in rheology 
    eqn += inner(0.5 * timestep * rheology_exp / params.T, evp.q) * dx
    eqn += inner(evp.q * zeta_exp * timestep / params.T, ep_dot_exp) * dx

    evp.assemble(eqn, evp.w1, evp.bcs, solver.srt_params)
    evp.u1, evp.s1 = evp.w1.split()

    diag = OutputDiagnostics(description="test 1", dirname=diagnostic_dirname)

    t = 0

    w = Function(evp.V, name="Exact Solution Vector").interpolate(v_exp)
    x = Function(evp.S, name="Exact Solution Tensor").interpolate(sigma_exp)

    while t < timescale - 0.5 * timestep:
        u0, s0 = evp.w0.split()
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        diag.dump(evp.w1, t=t)
        evp.dump(evp.u1, evp.s1, w, x, t=t)
        t += timestep
        evp.progress(t)
        print(Error.compute(evp.u1, w))
        #print(Error.compute(evp.s1, x))

    error_values.append(Error.compute(evp.u1, v_exp))

error_slope = float(format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f'))

print(error_slope)

plt.title('EVP Error Convergence')
plt.xlabel(r'Number of Triangles')
plt.ylabel(r'Error')
plt.loglog(number_of_triangles, error_values, '.', label='Gradient = {}'.format(error_slope))
plt.savefig(plot_dirname)





