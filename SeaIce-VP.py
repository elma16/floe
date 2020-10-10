from firedrake import *

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

n = 30
L = 500000

mesh = SquareMesh(n, n, L)

V = VectorFunctionSpace(mesh, "CR", 1)
W = FunctionSpace(mesh, "CR", 1)
U = MixedFunctionSpace((V, W, W))

# sea ice velocity
u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")

# mean height of sea ice
h_ = Function(W, name="Height")
h = Function(W, name="HeightNext")

# sea ice concentration
A_ = Function(W,name="Concentration")
A = Function(W,name="ConcentrationNext")

#test functions
v = TestFunction(V)
w = TestFunction(W)
q = TestFunction(W)

# defining the constants to be used in the sea ice momentum equation:


# the sea ice density
rho = 900

# gravity
g = 10

# Coriolis parameter
cor = 1.46 * 10 ** (-4)

# air density
rho_a = 1.3

# air drag coefficient
C_a = 1.2 * 10 ** (-3)

# water density
rho_w = 1026

# water drag coefficient
C_w = 5.5 * 10 ** (-3)

# ice strength parameter
P_star = 27.5 * 10 ** 3

# ice concentration parameter
C = 20

#  ellipse ratio
e = 2

x, y = SpatialCoordinate(mesh)

# initial conditions

ic = 10

u_.assign(ic)

u.assign(ic)

h_.assign(ic)

h.assign(ic)

A_.assign(ic)

A_.assign(ic)

timestep = 1 / n

# defining the functions that vary spatially and in time too i guess

# geostrophic wind

# geo_wind = as_vector([sin(pi * x)/L, 0])

geo_wind = as_vector([0,0])

# ocean current

ocean_curr = as_vector([2 * y / L - 1, 1 - 2 * x / L])

# VP rheology

# strain rate tensor, where grad(u) is the jacobian matrix of u
ep_dot = 1 / 2 * (grad(u) + transpose(grad(u)))

# deviatoric part of the strain rate tensor
ep_dot_prime = ep_dot - 1 / 2 * tr(ep_dot) * Identity(2)

# ice strength
P = P_star * h * exp(-C * (1 - A))

Delta_min = 2 * 10 ** (-9)

Delta = sqrt(Delta_min ** 2 + 2 * e ** (-2) * inner(ep_dot_prime, ep_dot_prime) + tr(ep_dot) ** 2)

# viscosities
zeta = P / (2 * Delta)
eta = zeta * e ** (-2)

# internal stress tensor
sigma = 2 * eta * ep_dot + (zeta - eta) * tr(ep_dot) * Identity(2) - P / 2 * Identity(2)

# solve the discretised sea ice momentum equation

# constructing the discretised weak form

# momentum equation
Lm = (inner(rho * h * (u - u_) / timestep - rho * h * cor * as_vector([u[1] - ocean_curr[1], ocean_curr[0] - u[0]])
           + rho_a * C_a * dot(geo_wind, geo_wind) * geo_wind + rho_w * C_w * dot(u - ocean_curr, u - ocean_curr) * (
                       ocean_curr - u), v) +
     inner(sigma, grad(v))) * dx

# balance laws

Lh = (inner((h - h_)/ timestep + div(u*h),q))*dx

La = (inner((A - A_)/ timestep + div(u*A),w))*dx

t = 0.0

hfile = File('h.pvd')
hfile.write(h_, time=t)
all_hs = []
end = 0.5
while (t <= end):
    solve(Lm == 0, u)
    u_.assign(u)
    #solve(Lh == 0, h)
    h_.assign(h)
    #solve(La == 0, A)
    A_.assign(A)
    t += timestep
    hfile.write(h_, time=t)
    print(t)

try:
  fig, axes = plt.subplots()
  plot(all_hs[-1], axes=axes)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)

