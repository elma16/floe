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
dumpfreq = 30 * hour
timescale = 24 * hour

dirname = path + "/u.pvd"

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
              'ocean_curr': ocean_curr,
              'geo_wind' : Constant(as_vector([0, 0])),
              'family':'CG',
              'stabilised': {'state': False , 'alpha': 10},
              'steady_state': False,
              'theta': 1}

conditions2 = {'bc': {'u': 0},
              'ic': {'u': 0, 'a' : x / length, 's' : as_matrix([[0, 0], [0, 0]])},
              'ocean_curr': ocean_curr,
              'geo_wind' : Constant(as_vector([0, 0])),
              'family':'CG',
              'stabilised': {'state': True , 'alpha': 10},
              'steady_state': False,
              'theta': 1}

timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq)
solver = SolverParameters()
params = SeaIceParameters()

evp = ElasticViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output,
                            params=params, solver_params=solver)
evp_stab = ElasticViscousPlastic(mesh=mesh, conditions=conditions2, timestepping=timestepping, output=output,
                                 params=params, solver_params=solver)
vp = ViscousPlastic(mesh=mesh, conditions=conditions, timestepping=timestepping, output=output, params=params,
                    solver_params=solver)
vp_stab = ViscousPlastic(mesh=mesh, conditions=conditions2, timestepping=timestepping, output=output, params=params,
                         solver_params=solver)



diag = OutputDiagnostics(description="EVP Energy Plot", dirname=diagnostic_dirname)

t = 0

while t < timescale - 0.5 * timestep:
    # solve first model
    u0, s0 = evp.w0.split()
    evp.solve(evp.usolver)
    evp.update(evp.w0, evp.w1)
    diag.dump(evp.u1, t)
    evp.dump(evp.u1, evp.s1, t=t)
    # solve second model
    u0, s0 = evp_stab.w0.split()
    evp_stab.solve(evp_stab.usolver)
    evp_stab.update(evp_stab.w0, evp_stab.w1)
    diag.dump(evp_stab.u1, t)
    evp_stab.dump(evp_stab.u1, evp_stab.s1, t=t)
    # solve third model
    vp.solve(vp.usolver)
    vp.update(vp.u0, vp.u1)
    diag.dump(vp.u1, t)
    vp.dump(vp.u1, t=t)
    # solve fourth (and last) model
    vp_stab.solve(vp_stab.usolver)
    vp_stab.update(vp_stab.u0, vp_stab.u1)
    diag.dump(vp_stab.u1, t)
    vp_stab.dump(vp_stab.u1, t=t)
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
