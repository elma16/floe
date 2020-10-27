from firedrake import *

"""
Transforming the mesh coordinates from a collection of right angled triangles on a square mesh
to a collection of equilateral triangles on a mesh.

This is achieved using a shear in the x-axis mapping (x,y) -> (x+1/2y,y) 
"""
n = 10

mesh = UnitSquareMesh(n,n)

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x+0.5*y, y]))
mesh.coordinates.assign(f)