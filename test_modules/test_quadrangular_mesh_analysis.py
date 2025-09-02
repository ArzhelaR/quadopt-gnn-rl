import unittest
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from environment.actions.quadrangular_actions import split_edge_ids, flip_edge_cw_ids
from view.mesh_plotter.mesh_plots import plot_mesh
from mesh_model.reader import read_gmsh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestMeshOldAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]]
        faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        cmap = Mesh(nodes,faces)
        qma = QuadMeshTopoAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = qma.global_score()
        self.assertEqual((0,0), (mesh_score, mesh_ideal_score) )

    def test_mesh_with_irregularities(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshTopoAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = qma.global_score()
        self.assertIsNot((0, 0), (mesh_score,mesh_ideal_score) )

    def test_is_valid_action(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshTopoAnalysis(cmap)

        #Boundary dart
        self.assertFalse(qma.isValidAction(20, 0))

        # Flip Clockwise test
        self.assertTrue(qma.isValidAction(3, 0))
        self.assertFalse(qma.isValidAction(27, 0))

        # Flip Counterclockwise test
        self.assertTrue(qma.isValidAction(3, 1))
        self.assertFalse(qma.isValidAction(27, 1))

        #Split test
        self.assertTrue(qma.isValidAction(0, 2))
        self.assertFalse(qma.isValidAction(27, 2))

        #Collapse test
        self.assertTrue(qma.isValidAction(0, 3))
        self.assertFalse(qma.isValidAction(27, 3))

        #Cleanup test action id = 4

        #All action test
        self.assertFalse(qma.isValidAction(27, 5) )
        flip_edge_cw_ids(qma,13,37)
        self.assertFalse(qma.isValidAction(66, 5))
        self.assertTrue(qma.isValidAction(9, 5))

        #One action test
        self.assertTrue(qma.isValidAction(0, 6))
        self.assertTrue(qma.isValidAction(9, 6))
        self.assertFalse(qma.isValidAction(27, 6))

        #Invalid action
        with self.assertRaises(ValueError):
            qma.isValidAction(0, 7)

    def test_isTruncated(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshTopoAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertFalse(qma.isTruncated(darts_list))

        nodes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        faces = [[0, 1, 3, 2]]
        cmap = Mesh(nodes, faces)
        qma = QuadMeshTopoAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertTrue(qma.isTruncated(darts_list))

if __name__ == '__main__':
    unittest.main()
