from seaice import *
from firedrake import *
from pathlib import Path

Path("./output/evp").mkdir(parents=True, exist_ok=True)

'''
Viscous plastic with ocean current forcing, and with zeta = zeta_max
'''

timestep = 0.1
dumpfreq = 1000
timescale = timestep * dumpfreq

stabilise = False
family = 'CG'

dirname = "./output/evp_vp/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(timescale, timestep, stabilise,
                                                                                       family)
title = "VP Plot"
diagnostic_dirname = "./output/evp_vp/evp_vp.nc"
plot_dirname = "./output/evp_vp/evp_vp_energy.png"

number_of_triangles = 35
length = 5 * 10 ** 5
mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
x, y = SpatialCoordinate(mesh)

ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
conditions = {'bc': {'u': 0},
              'ic': {'u': 0, 'a' : x / length},
              'ocean_curr': ocean_curr}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

vp = ViscousPlasticHack(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output, params=params, solver_params=solver, stabilised=stabilise, family=family)

diag = OutputDiagnostics(description="EVP-VP Test", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    vp.solve(vp.usolver)
    vp.update(vp.u0, vp.u1)
    vp.dump(vp.u1, t=t)
    t += timestep
    vp.progress(t)






