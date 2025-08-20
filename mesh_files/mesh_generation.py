from mesh_model.random_quadmesh import random_mesh
from mesh_model.reader import read_gmsh

for i in range(100):
    #mesh = read_gmsh("simple_quad.msh")
    mesh_rd = random_mesh()