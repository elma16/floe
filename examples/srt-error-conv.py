from seaice import *
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "./output/srt-error-conv"
Path(path).mkdir(parents=True, exist_ok=True)

timestep = 1
dumpfreq = 10 ** 6
timescale = 10

zero = Constant(0)
zero_vector = Constant(as_vector([0, 0]))

dirname = path + "/u.pvd"
plot_dirname = path + "/srt-conv.png"

number_of_triangles = [5, 10, 20, 40, 100]

error_values = []

length = 5 * 10 ** 5
pi_x = pi / length
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters(rho=1, rho_a=zero, C_a=zero, rho_w=zero, C_w=zero, cor=zero)

for values in number_of_triangles:
    mesh = SquareMesh(values, values, length)
    x, y = SpatialCoordinate(mesh)
    v_exp = as_vector([-sin(pi_x * x) * sin(pi_x * y), -sin(pi_x * x) * sin(pi_x * y)])

    ic = {'u': v_exp, 'a' : 1, 'h' : 1}
    conditions = Conditions(ic=ic, theta=1)
    srt = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                         solver_params=solver)

    zeta = srt.zeta(srt.h, srt.a, params.Delta_min)
    sigma = zeta * srt.strain(grad(srt.u1))
    sigma_exp = zeta * srt.strain(grad(v_exp))

    eqn = srt.momentum_equation(srt.h, srt.u1, srt.u0, srt.p, sigma, params.rho, zero_vector, conditions.ocean_curr,
                            params.rho_a, params.C_a, params.rho_w, params.C_w, conditions.geo_wind, params.cor, timestep)
    eqn += timestep * inner(div(sigma_exp), srt.p) * dx

    srt.assemble(eqn,srt.u1,srt.bcs,solver.srt_params)

    t = 0

    while t < timescale - 0.5 * timestep:
        srt.solve(srt.usolver)
        srt.update(srt.u0, srt.u1)
        t += timestep

    error_values.append(Error.compute(srt.u1, v_exp))

error_slope = float(format(np.polyfit(np.log(number_of_triangles), np.log(error_values), 1)[0], '.3f'))

print(error_slope)

plt.title('Strain Rate Tensor Error Convergence')
plt.xlabel(r'Number of Triangles')
plt.ylabel(r'Error')
plt.loglog(number_of_triangles, error_values, '.', label='Gradient = {}'.format(error_slope))
plt.savefig(plot_dirname)


