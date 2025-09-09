from view.mesh_plotter.mesh_plots import plot_dataset
from mesh_model.reader import read_dataset

dataset = read_dataset("../../training/dataset/training_dataset")
plot_dataset(dataset)