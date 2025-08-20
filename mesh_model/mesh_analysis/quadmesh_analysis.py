import math
from doctest import debug

import numpy as np
import matplotlib.pyplot as plt
import warnings

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.global_mesh_analysis import GlobalMeshAnalysis, NodeAnalysis
from view.mesh_plotter.mesh_plots import plot_mesh

FLIP_CW = 0 # flip clockwise
FLIP_CCW = 1 # flip counterclockwise
SPLIT = 2
COLLAPSE = 3
CLEANUP = 4
TEST_ALL = 5 # test if all actions are valid
ONE_VALID = 6 # test if at least one action is valid

class QuadMeshAnalysis(GlobalMeshAnalysis):
    """
    The base of quadrangular mesh analysis
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)
        #If initial scores and adjacency have not already been set
        if self.mesh.nodes[0,4] == -99:
            self.set_adjacency()
            self.set_scores()

    def isValidAction(self, dart_id: int, action: int) -> bool:
        pass

    def set_adjacency(self) -> None:
        i = 0
        for n_info in self.mesh.nodes:
            if n_info[2] >= 0:
                n = Node(self.mesh, i)
                na = NodeAnalysis(n)
                if na.on_boundary():
                    angle = na.get_boundary_angle()
                    ideal_adj = max(round(angle / 90) + 1, 2)
                    n.set_ideal_adjacency(ideal_adj)
                else:
                    n.set_ideal_adjacency(4)
            i += 1

    def set_scores(self) -> None:
        i = 0
        for n_info in self.mesh.nodes:
            if n_info[2] >= 0:
                n = Node(self.mesh, i)
                na = NodeAnalysis(n)
                s = na.score_calculation()
                n.set_score(s)
            i += 1

    def get_adjacent_nodes(self, d:Dart)->list:
        """
        We collect all the adjacent nodes of the dart. The dart d is linked to node n1. We want to look at adjacent edges of nodes n1 and n3 (opposite node of n1 in the quad)
        :param d:
        :return:
        """
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        n3 = d11.get_node()


        adj_nodes = []
        nodes_coord = []

        d = d1.get_beta(2)
        # d12 = d1.get_beta(2)
        # d = d12.get_beta(1)
        n = d.get_node()

        while n != n1:
            adj_nodes.append(n)
            nodes_coord.append([n.x(), n.y()])
            d1 = d.get_beta(1)
            d = d1.get_beta(2)
            if d is None:
                raise ValueError("Error on getting kernel")
            n = d.get_node()

        d2 = d.get_beta(2)
        d = (d2.get_beta(1)).get_beta(2)
        n = d.get_node()
        while n != n2:
            adj_nodes.append(n)
            nodes_coord.append([n.x(), n.y()])
            d1 = d.get_beta(1)
            d = d1.get_beta(2)
            n = d.get_node()
        return nodes_coord

class QuadMeshTopoAnalysis(QuadMeshAnalysis):
    """
    Quadmesh old analysis
    """

    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)

    def isValidAction(self, dart_id: int, action: int) -> bool:
        """
        Test if an action is valid. You can select the ype of action between {flip clockwise, flip counterclockwise, split, collapse, cleanup, all action, one action no matter wich one}.
        :param dart_id: a dart on which to test the action
        :param action: an action type
        :return:
        """
        d = Dart(self.mesh, dart_id)
        if d.get_beta(2) is None:
            return False
        elif action == FLIP_CW:
            return self.isFlipCWOk(d)
        elif action == FLIP_CCW:
            return self.isFlipCCWOk(d)
        elif action == SPLIT:
            return self.isSplitOk(d)
        elif action == COLLAPSE:
            return self.isCollapseOk(d)
        elif action == CLEANUP:
            return self.isCleanupOk(d)
        elif action == TEST_ALL:
            return self.isFlipCCWOk(d) and self.isFlipCWOk(d) and self.isSplitOk(d) and self.isCollapseOk(d)
        elif action == ONE_VALID:
            topo_flip_ccw= self.isFlipCCWOk(d)
            if topo_flip_ccw:
                return True
            topo_flip_cw= self.isFlipCWOk(d)
            if topo_flip_cw:
                return True
            topo_split= self.isSplitOk(d)
            if topo_split:
                return True
            topo_collapse = self.isCollapseOk(d)
            if topo_collapse:
                return True
            return False
        else:
            raise ValueError("No valid action")


    def isFlipCCWOk(self, d: Dart) -> bool:
        topo = True

        # if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n5_analysis = NodeAnalysis(n5)
        n3_analysis = NodeAnalysis(n3)
        # if degree will not too high
        if not n5_analysis.test_degree() or not n3_analysis.test_degree():
            topo = False
            return topo

        # if two faces share two edges
        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo

        return topo

    def isFlipCWOk(self, d: Dart) -> bool:
        topo = True
        # if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n4_analysis = NodeAnalysis(n4)
        n6_analysis = NodeAnalysis(n6)
        if not n4_analysis.test_degree() or not n6_analysis.test_degree():
            topo = False
            return topo

        # if two faces share two edges
        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo
        return topo


    def isSplitOk(self, d: Dart) -> bool:
        topo = True
        if d.get_beta(2) is None:
            topo = False
            return topo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n4_analysis = NodeAnalysis(n4)
        n2_analysis = NodeAnalysis(n2)
        if not n4_analysis.test_degree() or not n2_analysis.test_degree():
            topo = False
            return topo

        # if two faces share two edges
        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo

        return topo


    def isCollapseOk(self, d: Dart) -> bool:
        topo = True
        if d.get_beta(2) is None:
            topo = False
            return topo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n1_analysis = NodeAnalysis(n1)
        n3_analysis = NodeAnalysis(n3)

        if n1_analysis.on_boundary():
            topo = False
            return topo

        if (n3_analysis.degree() +n1_analysis.degree()-2) > 10:
            topo = False
            return topo

        adjacent_faces_lst = []
        f1 = d2.get_face()
        adjacent_faces_lst.append(f1.id)
        d12 = d1.get_beta(2)
        if d12 is not None:
            f2 = d12.get_face()
            adjacent_faces_lst.append(f2.id)
        d112 = d11.get_beta(2)
        if d112 is not None:
            f3 = d112.get_face()
            adjacent_faces_lst.append(f3.id)
        d1112 = d111.get_beta(2)
        if d1112 is not None:
            f4 = d1112.get_face()
            adjacent_faces_lst.append(f4.id)

        # Check that there are no adjacent faces in common
        if len(adjacent_faces_lst) != len(set(adjacent_faces_lst)):
            topo = False
            return topo

        return topo


    def isCleanupOk(self, d: Dart) -> (bool, bool):
        topo = True
        # plot_mesh(d.mesh, debug=True)
        if d.get_beta(2) is None:
            topo = False
        mesh = d.mesh
        parallel_darts = mesh.find_parallel_darts(d)
        faces = []
        nodes_from = []
        nodes_to = []
        for dp in parallel_darts:
            f = dp.get_face()
            n_from = dp.get_node()
            n_to = (dp.get_beta(1)).get_node()
            # If the dart is not a boundary dart but the two nodes are on boundary we can't cleanup
            if dp.get_beta(2) is not None:
                na_from = NodeAnalysis(n_from)
                na_to = NodeAnalysis(n_to)
                if na_from.on_boundary() and na_to.on_boundary():
                    return False, False

            nodes_from.append(n_from)
            nodes_to.append(n_to)
            if f.id not in faces:
                faces.append(f.id)
            else:
                topo = False
                return topo, False
            d111 = (( dp.get_beta(1)).get_beta(1)).get_beta(1)
            if d111.get_beta(2) is None:
                topo = False
                return topo, False

        # The last dart lies on the boundary, so it's not included in the list of parallel darts.
        # As a result, its associated nodes were not retrieved automatically, so we handle them manually here.
        last_dart = parallel_darts[-1]
        ld1 = last_dart.get_beta(1)
        ld11 = ld1.get_beta(1)
        ld111 = ld11.get_beta(1)
        last_node_from = ld111.get_node()
        last_node_to = ld11.get_node()
        nodes_from.append(last_node_from)
        nodes_to.append(last_node_to)
        removable_n_from = self.are_nodes_removables(nodes_from)
        removable_n_to = self.are_nodes_removables(nodes_to)

        if not removable_n_from and not removable_n_to:
            topo = False
            return topo, False
        else:
            return topo, removable_n_from

    def are_nodes_removables(self, nodes_from):
        if len(nodes_from) != len(set(n.id for n in nodes_from)):
            return False
        else:
            for n in nodes_from:
                na = NodeAnalysis(n)
                if na.on_boundary() and n.get_ideal_adjacency() != 3:
                    return False
            return True

    def isTruncated(self, darts_list)-> bool:
        for d_id in darts_list:
            if self.isValidAction(d_id, 4):
                return False
        return True