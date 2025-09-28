from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from view.mesh_plotter.mesh_plots import plot_dataset, plot_mesh
from mesh_model.reader import read_dataset, read_gmsh

dataset = read_dataset("../../training/dataset/training_dataset")
plot_dataset(dataset)
# cmap = read_gmsh("../../mesh_files/mesh_019.msh")
# ma = QuadMeshTopoAnalysis(cmap)
# plot_mesh(ma.mesh)
