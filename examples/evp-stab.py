from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp-stabilised"
Path(path).mkdir(parents=True, exist_ok=True)

'''
TEST 2 : EVP 

Find the value of the stabilisation parameter to obtain a stabilised velocity plot
'''

timestep = 0.1
dumpfreq = 10 ** 4
timescale = timestep * dumpfreq

stabilise = True
family = 'CR'

for alpha in [10,20,50,100,500,1000]:
    dirname = path + "/u_timescale={}_timestep={}_stabilised={}_value={}.pvd".format(timescale, timestep, stabilise, alpha)
                                                                            
    title = "EVP Plot"
    diagnostic_dirname = path + "evp.nc"
    plot_dirname = path + "evp_energy.png"

    number_of_triangles = 35
    length = 5 * 10 ** 5
    mesh = SquareMesh(number_of_triangles, number_of_triangles, length)
    x, y = SpatialCoordinate(mesh)

    ocean_curr = as_vector([0.1 * (2 * y - length) / length, -0.1 * (length - 2 * x) / length])
    conditions = {'bc': {'u': 0},
                  'ic': {'u': 0, 'a' : x / length, 's' : as_matrix([[0, 0], [0, 0]])},
                  'ocean_curr': ocean_curr}

    timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
    output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
    solver = SolverParameters()
    params = SeaIceParameters()


    evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output, params=params, solver_params=solver, stabilised=stabilise, family=family,theta=1,steady_state=False, alpha = alpha)

    diag = OutputDiagnostics(description="EVP Test", dirname=diagnostic_dirname)

    t = 0

    while t < timescale - 0.5 * timestep:
        u0, s0 = evp.w0.split()
        evp.solve(evp.usolver)
        evp.update(evp.w0, evp.w1)
        evp.dump(evp.u1, evp.s1, t=t)
        t += timestep
        evp.progress(t)

