from view.mesh_plotter.mesh_plots import plot_dataset, plot_mesh
from mesh_model.reader import read_gmsh
from mesh_model.random_trimesh import random_mesh

mesh = read_gmsh('../../mesh_files/simple_quad.msh')
plot_mesh(mesh)