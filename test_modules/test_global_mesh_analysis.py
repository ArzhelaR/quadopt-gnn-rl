import math
import unittest
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Node
from mesh_model.reader import read_gmsh
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from mesh_model.mesh_analysis.global_mesh_analysis import NodeAnalysis
from view.mesh_plotter.mesh_plots import plot_mesh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestGlobalMeshAnalysis(unittest.TestCase):

    def testOnBoundary(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        ma = QuadMeshTopoAnalysis(cmap)
        plot_mesh(cmap, debug=True)

        n_to_test = Node(cmap, 0)
        na = NodeAnalysis(n_to_test)
        self.assertTrue(na.on_boundary())

if __name__ == '__main__':
    unittest.main()