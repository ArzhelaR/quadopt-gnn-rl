from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from view.mesh_plotter.mesh_plots import plot_dataset, plot_mesh, save_dataset_plot
from mesh_model.reader import read_dataset, read_gmsh, read_json

# dataset = read_dataset("../../training/dataset/results/bunny-3darts")
# #plot_dataset(dataset)
# save_dataset_plot(dataset, "Supplementary_plots/.png")
# print("File saved")
cmap = read_json("../../mesh_files/L_config.json") # ../../training/dataset/results/imr3_results/mesh_0.json
ma = QuadMeshTopoAnalysis(cmap)
plot_mesh(ma.mesh, scores=True)
