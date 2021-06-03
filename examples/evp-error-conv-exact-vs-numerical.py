from seaice import *
from firedrake import *
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

path = "./output/evp-error-conv_ex-vs-num"
Path(path).mkdir(parents=True, exist_ok=True)

"""
TEST 2 : EVP Error Convergence - exact vs. numerical

Manufactured solutions.
Coriolis force neglected, no forcing due to wind. 
Forcing due to ocean is present.
Advection turned off.
Boundary conditions : u = 0
Initial conditions : h = 1, A = x / L
Domain is a 500km x 500km square.

Exact : project initial condition 
Numerical : interpolate intitial condtion

Compute the error between these two.
"""

timestep = 1
dumpfreq = 10 ** 3
timescale = 10

zero = Constant(0)

norm_type = "L2"

title = "EVP Plot"
diagnostic_dirname = path + "/evp.nc"
plot_dirname = path + "/evp_error_timescale={}_timestep={}_{}.png".format(
    timescale, timestep, norm_type
)
dirname = path + "/u_timescale={}_timestep={}.pvd".format(timescale, timestep)

length = 5 * 10 ** 5
pi_x = pi / length

number_of_triangles = [5, 10, 20, 40, 80]

error_values = []

stabilised = {"state": False, "alpha": 1}
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters(rho_a=zero, C_a=zero, cor=zero)

for values in number_of_triangles:
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector(
        [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)]
    )
    sigma_exp = as_matrix(
        [
            [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
            [-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)],
        ]
    )

    ocean_curr = as_vector(
        [0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length]
    )

    ic = {"u": v_exp, "a": 1, "h": 1, "s": sigma_exp}
    
    conditions_ex = Conditions(
        ic=ic, ocean_curr=ocean_curr, stabilised=stabilised, family="CR", exact=True
    )

    conditions_num = Conditions(
        ic=ic, ocean_curr=ocean_curr, stabilised=stabilised, family="CR", exact=False
    )

    evp_num = ElasticViscousPlastic(
        mesh=mesh,
        conditions=conditions_num,
        timestepping=timestepping,
        output=output,
        params=params,
        solver_params=solver,
    )
    
    evp_ex = ElasticViscousPlastic(
        mesh=mesh,
        conditions=conditions_ex,
        timestepping=timestepping,
        output=output,
        params=params,
        solver_params=solver,
    )

    u1_num, s1_num = split(evp_num.w1)
    u0_num, s0_num = split(evp_num.w0)

    theta = 0.5
    uh_num = (1 - theta) * u0_num + theta * u1_num
    sh_num = (1 - theta) * s0_num + theta * s1_num

    eqn_num = inner(params.rho * evp_num.h * (u1_num - u0_num), evp_num.p) * dx
    eqn_num += timestep * inner(sh_num, grad(evp_num.p)) * dx
    eqn_num -= (
        timestep
        * inner(
            params.rho_w
            * params.C_w
            * sqrt(dot(ocean_curr - uh_num, ocean_curr - uh_num))
            * (ocean_curr - uh_num),
            evp_num.p,
        )
        * dx
    )

    # source terms in momentum equation
    eqn_num += timestep * inner(div(sigma_exp), evp_num.p) * dx
    eqn_num += (
        timestep
        * inner(
            params.rho_w
            * params.C_w
            * sqrt(dot(ocean_curr - v_exp, ocean_curr - v_exp))
            * (ocean_curr - v_exp),
            evp_num.p,
        )
        * dx
    )

    zeta_num = evp_num.zeta(evp_num.h, evp_num.a, evp_num.delta(v_exp))
    ep_dot_num = evp_num.strain(grad(v_exp))
    rheology_num = params.e ** 2 * sigma_exp + Identity(2) * 0.5 * (
        (1 - params.e ** 2) * tr(sigma_exp) + evp_num.Ice_Strength(evp_num.h, evp_num.a)
    )
    zeta = evp_num.zeta(evp_num.h, evp_num.a, evp_num.delta(uh_num))
    
    eqn_num += inner(s1_num - s0_num + 0.5 * timestep * evp_num.rheology / params.T, evp_num.q) * dx
    eqn_num -= inner(evp_num.q * zeta * timestep / params.T, evp_num.ep_dot) * dx
    
    # source terms in rheology
    eqn_num -= inner(0.5 * timestep * rheology_num / params.T, evp_num.q) * dx
    eqn_num += inner(evp_num.q * zeta_num * timestep / params.T, ep_dot_num) * dx

    evp_num.assemble(eqn_num, evp_num.w1, evp_num.bcs, solver.srt_params)

    # ---

    u1_ex, s1_ex = split(evp_ex.w1)
    u0_ex, s0_ex = split(evp_ex.w0)

    theta = 0.5
    uh_ex = (1 - theta) * u0_ex + theta * u1_ex
    sh_ex = (1 - theta) * s0_ex + theta * s1_ex

    eqn_ex = inner(params.rho * evp_ex.h * (u1_ex - u0_ex), evp_ex.p) * dx
    eqn_ex += timestep * inner(sh_ex, grad(evp_ex.p)) * dx
    eqn_ex -= (
        timestep
        * inner(
            params.rho_w
            * params.C_w
            * sqrt(dot(ocean_curr - uh_ex, ocean_curr - uh_ex))
            * (ocean_curr - uh_ex),
            evp_ex.p,
        )
        * dx
    )

    # source terms in momentum equation
    eqn_ex += timestep * inner(div(sigma_exp), evp_ex.p) * dx
    eqn_ex += (
        timestep
        * inner(
            params.rho_w
            * params.C_w
            * sqrt(dot(ocean_curr - v_exp, ocean_curr - v_exp))
            * (ocean_curr - v_exp),
            evp_ex.p,
        )
        * dx
    )

    zeta_ex = evp_ex.zeta(evp_ex.h, evp_ex.a, evp_ex.delta(v_exp))
    ep_dot_ex = evp_ex.strain(grad(v_exp))
    rheology_ex = params.e ** 2 * sigma_exp + Identity(2) * 0.5 * (
        (1 - params.e ** 2) * tr(sigma_exp) + evp_ex.Ice_Strength(evp_ex.h, evp_ex.a)
    )
    zeta_ex = evp_ex.zeta(evp_ex.h, evp_ex.a, evp_ex.delta(uh_ex))
    
    eqn_ex += inner(s1_ex - s0_ex + 0.5 * timestep * evp_ex.rheology / params.T, evp_ex.q) * dx
    eqn_ex -= inner(evp_ex.q * zeta_ex * timestep / params.T, evp_ex.ep_dot) * dx
    
    # source terms in rheology
    eqn_ex -= inner(0.5 * timestep * rheology_ex / params.T, evp_ex.q) * dx
    eqn_ex += inner(evp_ex.q * zeta_ex * timestep / params.T, ep_dot_ex) * dx

    evp_ex.assemble(eqn_ex, evp_ex.w1, evp_ex.bcs, solver.srt_params)

    # ---

    diag = OutputDiagnostics(description="test 1", dirname=diagnostic_dirname)

    t = 0

    u1_num, s1_num = evp_num.w1.split()
    u1_ex, s1_ex = evp_ex.w1.split()

    evp_num.dump(u1_num, s1_num, t=0)
    evp_ex.dump(u1_ex, s1_ex, t=0)

    while t < timescale - 0.5 * timestep:
        u0_num, s0_num = evp_num.w0.split()
        u0_ex, s0_ex = evp_ex.w0.split()
        evp_num.solve(evp_num.usolver)
        evp_ex.solve(evp_ex.usolver)
        evp_num.update(evp_num.w0, evp_num.w1)
        evp_ex.update(evp_ex.w0, evp_ex.w1)
        diag.dump(evp_ex.w1, t=t)
        diag.dump(evp_num.w1, t=t)
        t += timestep
        evp_num.dump(u1_num, s1_num, t=t)
        evp_ex.dump(u1_ex, s1_ex, t=t)
        evp_num.progress(t)
        evp_ex.progress(t)
        print(Error.compute(u1_num, u1_ex, norm_type))
        # print(Error.compute(evp.s1, x))

    error_values.append(Error.compute(u1_num, u1_ex, norm_type))

h = [sqrt(2) * length / x for x in number_of_triangles]
error_slope = float(format(np.polyfit(np.log(h), np.log(error_values), 1)[0], ".3f"))

print(error_slope)

plt.title("EVP Error Convergence")
plt.xlabel(r"h")
plt.ylabel(r"{} Error".format(norm_type))
plt.loglog(h, error_values, ".", label="Gradient = {}".format(error_slope))
plt.savefig(plot_dirname)
