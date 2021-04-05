from seaice import *
from firedrake import *
from pathlib import Path

path = "./output/evp-energy-plot"
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

hour = 60*60
timestep = 0.1
dumpfreq = hour
timescale = 24 * hour

stabilise = False
family = 'CR'

dirname = path + "/u_timescale={}_timestep={}_stabilised={}_family={}.pvd".format(timescale, timestep, stabilise,
                                                                                       family)
fig5a_title = "Figure 5 a)"
fig5b_title = "Figure 5 b)"
fig5c_title = "Figure 5 c)"
diagnostic_dirname = path + "/evp_energy_plot.nc"
fig5a_dirname = path + "/fig5a.png"
fig5b_dirname = path + "/fig5b.png"
fig5c_dirname = path + "/fig5c.png"

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

evp = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                            params=params, solver_params=solver, stabilised=False, family=family, theta=1,
                            steady_state=False,alpha=0)
evp_stab = ElasticViscousPlastic(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                                 params=params, solver_params=solver, stabilised=True, family=family, theta=1,
                                 steady_state=False, alpha=1000)
vp = ViscousPlasticHack(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                        params=params, solver_params=solver, stabilised=False, family=family)
vp_stab = ViscousPlasticHack(mesh=mesh, length=length, conditions=conditions, timestepping=timestepping, output=output,
                             params=params, solver_params=solver, stabilised=True, family=family)



diag = OutputDiagnostics(description="EVP Energy Plot", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    # solve first model
    u0, s0 = evp_ustab.w0.split()
    evp_ustab.solve(evp_ustab.usolver)
    evp_ustab.update(evp_ustab.w0, evp_ustab.w1)
    diag.dump(evp_ustab.u1, t)
    evp_ustab.dump(evp_ustab.u1, evp_ustab.s1, t=t)
    # solve second model
    u0, s0 = evp_ustab.w0.split()
    evp_ustab.solve(evp_ustab.usolver)
    evp_ustab.update(evp_ustab.w0, evp_ustab.w1)
    diag.dump(evp_ustab.u1, t)
    evp_ustab.dump(evp_ustab.u1, evp_ustab.s1, t=t)
    # solve third model
    u0, s0 = evp_ustab.w0.split()
    evp_ustab.solve(evp_ustab.usolver)
    evp_ustab.update(evp_ustab.w0, evp_ustab.w1)
    diag.dump(evp_ustab.u1, t)
    evp_ustab.dump(evp_ustab.u1, evp_ustab.s1, t=t)
    # solve fourth (and last) model
    u0, s0 = evp_ustab.w0.split()
    evp_ustab.solve(evp_ustab.usolver)
    evp_ustab.update(evp_ustab.w0, evp_ustab.w1)
    diag.dump(evp_ustab.u1, t)
    evp_ustab.dump(evp_ustab.u1, evp_ustab.s1, t=t)
    # update time and progress
    t += timestep
    vp_stab.progress(t)
    
fig5a = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=fig5a_dirname,
                  timestepping=timestepping, title=fig5a_title)
fig5b = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=fig5b_dirname,
                  timestepping=timestepping, title=fig5b_title)
fig5c = Plotter(dataset_dirname=diagnostic_dirname, diagnostic='energy', plot_dirname=fig5c_dirname,
                  timestepping=timestepping, title=fig5c_title)

fig5a.plot(plot=plot)
