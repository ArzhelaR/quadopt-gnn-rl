from view.mesh_plotter.mesh_plots import plot_dataset, plot_mesh
from mesh_model.reader import read_dataset, read_gmsh

dataset = read_dataset("../../training/dataset/test_dataset")
plot_dataset(dataset)
