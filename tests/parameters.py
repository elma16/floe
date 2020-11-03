from firedrake import *
"""
All the parameters required to solve the sea ice momentum equation
"""
#dimension of the mesh
L = 500000

#box test mesh dimensions
L2 = 1000000

# the sea ice density
rho = Constant(900)

# Coriolis parameter
cor = Constant(1.46 * 10 ** (-4))

# air density
rho_a = Constant(1.3)

# air drag coefficient
C_a = Constant(1.2 * 10 ** (-3))

# water density
rho_w = Constant(1026)

# water drag coefficient
C_w = Constant(5.5 * 10 ** (-3))

# ice strength parameter
P_star = Constant(27.5 * 10 ** 3)

# ice concentration parameter
C = Constant(20)

#  ellipse ratio
e = Constant(2)

Delta_min = Constant(2 * 10 ** (-9))

#tuning parameter
T = 100


#mEVP
alpha = Constant(500)
beta = Constant(500)