from seaice import *
from firedrake import *
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

path = "./output/mk/figure5"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP

File reproducing figure 5 from Mehlmann and Korn (2021)

Fig 5 a) Energy vs. t (0-24h)
compare evp and vp unstabilised and stabilised
Fig 5 b) Energy vs. t (0-24h)
evp unstabilised and stabilised applied to resolution (10km, 5km, 2.5km)
Fig 5 c) Energy (log) vs. resolution (10km, 5km, 2.5km)
'''

timestep = 1
dumpfreq = 10**10
timescale = 24*60*60

dirname = path + "/u.pvd"

fig5c_title = "Figure 5 c)"
d_dirname1 = path + "/evp_energy2.nc"
d_dirname2 = path + "/evp_stab_energy2.nc"
fig5c_dirname = path + "/fig5c.png"


for triangles in [10, 20, 30]:
    number_of_triangles = triangles
    length = 5 * 10 ** 5
    mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
    x, y = SpatialCoordinate(mesh)

    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])

    ic =  {'u': 0, 'a': x / length, 'h': 1, 's': as_matrix([[0, 0], [0, 0]])}
    conditions = Conditions(theta=0.5, ocean_curr=ocean_curr, ic=ic)
    stabilised =  {'state': True , 'alpha': 10}
    conditions_stab = Conditions(theta=0.5, ocean_curr=ocean_curr, stabilised=stabilised, ic=ic)
        
    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()

    evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params, solver_params=solver)

    evp_stab = ElasticViscousPlastic(mesh=mesh, conditions=conditions_stab, timestepping=timestepping, output=output, params=params, solver_params=solver)
    
    t = 0

    diag1 = OutputDiagnostics(description="EVP Energy", dirname=d_dirname1)
    diag2 = OutputDiagnostics(description="EVP Stabilised Energy", dirname=d_dirname2)

    while t < timescale - 0.5 * timestep:
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        evp.dump(evp.u1, evp.s1, t=t)
        evp_stab.solve(evp_stab.usolver)
        evp_stab.update(evp_stab.w0, evp_stab.w1)
        evp_stab.dump(evp_stab.u1, evp_stab.s1, t=t)
        t += timestep
        evp.progress(t)
        
    diag1.dump(evp.u1, t)
    diag2.dump(evp_stab.u1, t)

    # fig 5c

    dataset1 = Dataset(d_dirname1, mode='r')
    yaxis1 = dataset1.variables['energy'][:]
    dataset1.close()
    dataset2 = Dataset(d_dirname2, mode='r')
    yaxis2 = dataset2.variables['energy'][:]
    dataset2.close()
    plt.semilogy(triangles,yaxis1,'o-',label='{} triangles'.format(triangles))
    plt.semilogy(triangles,yaxis2,'o-',label='{} triangles stabilised'.format(triangles))
    plt.ylabel(r'Energy of solution')
    plt.xlabel(r'Mesh Size')
    plt.title(fig5c_title)
    plt.legend(loc='best')

plt.savefig(fig5c_dirname)




