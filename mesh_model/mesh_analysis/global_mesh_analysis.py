import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, degrees, acos, atan2
from abc import ABC, abstractmethod

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
#from view.mesh_plotter.mesh_plots import plot_mesh


class NodeAnalysis:
    def __init__(self, n: Node):
        self.n = n

    def score_calculation(self) -> int:
        """
        Function to calculate the irregularity of a node in the mesh.
        :param n: a node in the mesh.
        :return: the irregularity of the node
        :raises ValueError: if the node is associated to no dart
        """

        d = self.n.get_dart()
        if d.mesh.dart_info[d.id,0] < 0:
            raise ValueError("No existing dart")

        adjacency = self.degree()
        ideal_adjacency =self.n.get_ideal_adjacency()

        return ideal_adjacency - adjacency

    def get_angle(self, d1: Dart, d2: Dart) -> float:
        """
        Function to calculate the angle of the boundary at the node n.
        The angle is named ABC and node self.n is at point A.
        :param d1: the first boundary dart.
        :param d2: the second boundary dart.
        :return: the angle (degrees)
        """
        if d1.get_node() == self.n:
            A = self.n
            B = d1.get_beta(1).get_node()
            C = d2.get_node()

        else:
            A = self.n
            B = d2.get_beta(1).get_node()
            C = d1.get_node()
            if d2.get_node() != A:
                raise ValueError("Angle error")

        vect_AB = (B.x() - A.x(), B.y() - A.y())
        vect_AC = (C.x() - A.x(), C.y() - A.y())

        dot = vect_AB[0] * vect_AC[0] + vect_AB[1] * vect_AC[1]

        # cross product
        cross = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]

        angle_rad = atan2(cross, dot)
        angle = degrees(angle_rad) % 360
        if np.isnan(angle):
            raise ValueError("Angle error")
        return angle

    def get_boundary_angle(self) -> float:
        """
        Calculate the boundary angle of a node in the mesh.
        :return: the boundary angle (degrees)
        """
        adj_darts_list = self.adjacent_darts()
        boundary_darts = []
        for d in adj_darts_list:
            d_twin = d.get_beta(2)
            if d_twin is None:
                boundary_darts.append(d)
        # if len(boundary_darts) > 2 : # or len(boundary_darts) < 2
        #     plot_mesh(self.n.mesh)
        #     raise ValueError("Boundary error")
        angle = self.get_angle(boundary_darts[0], boundary_darts[1])
        return angle

    def on_boundary(self) -> bool:
        """
        Test if the node self.n is on boundary.
        :return: True if the node n is on boundary, False otherwise.
        """
        adj_darts_list = self.adjacent_darts()
        for d in adj_darts_list:
            d_twin = d.get_beta(2)
            if d_twin is None:
                return True
        return False

    def adjacent_darts(self) -> list[Dart]:
        """
        Function that retrieve the adjacent darts of node n.
        :return: the list of adjacent darts
        """
        adj_darts = []
        m = self.n.mesh
        for d_info in m.active_darts():
            d = Dart(m, d_info[0])
            d_nfrom = d.get_node()
            d_nto = d.get_beta(1)
            if d_nfrom == self.n and d not in adj_darts:
                adj_darts.append(d)
            if d_nto.get_node() == self.n and d not in adj_darts:
                adj_darts.append(d)
        return adj_darts

    def adjacent_faces_id(self) -> list[int]:
        adj_darts = self.adjacent_darts()
        adj_faces = []
        for d in adj_darts:
            f = d.get_face()
            if f.id not in adj_faces:
                adj_faces.append(f.id)
        return adj_faces

    def degree(self, mesh_before=None) -> int:
        """
        Function to calculate the degree of a node in the mesh.
        :return: the degree of the node
        """
        adj_darts_list = self.adjacent_darts()
        adjacency = 0
        b = self.on_boundary()
        boundary_darts = []
        for d in adj_darts_list:
            d_twin = d.get_beta(2)
            if d_twin is None and b:
                adjacency += 1
                boundary_darts.append(d)
            else:
                adjacency += 0.5
        if adjacency != int(adjacency):
            plot_mesh(mesh_before, debug=True)
            plot_mesh(self.n.mesh, debug=True)
            raise ValueError("Adjacency error")
        return adjacency

    def test_degree(self) -> bool:
        """
        Verify that the degree of a vertex is lower than 10
        :return: True if the degree is lower than 10, False otherwise
        """
        if self.degree() >= 10:
            return False
        else:
            return True


class GlobalMeshAnalysis(ABC):
    """
    The base of mesh analysis
    :param mesh: A mesh to analise
    """
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def set_adjacency(self):
        pass

    def set_scores(self):
        pass

    def global_score(self, mesh_before = None):
        """
        Calculate the overall mesh score. The mesh cannot achieve a better score than the ideal one.
        And the current score is the mesh score.
        :return: 4 return: a list of the nodes score, the current mesh score and the ideal mesh score, and the adjacency
        """
        mesh_ideal_score = 0
        mesh_score = 0
        nodes_score = []
        for i in range(len(self.mesh.nodes)):
            if self.mesh.nodes[i, 2] >= 0:
                n_id = i
                node = Node(self.mesh, n_id)
                n_a = NodeAnalysis(node)
                n_score = node.get_score()
                nodes_score.append(n_score)
                mesh_ideal_score += n_score
                mesh_score += abs(n_score)
            else:
                nodes_score.append(0)
        return nodes_score, mesh_score, mesh_ideal_score

    def get_boundary_darts(self) -> list[Dart]:
        """
        Find all boundary darts
        :return: a list of all boundary darts
        """
        boundary_darts = []
        for d_info in self.mesh.active_darts():
            d = Dart(self.mesh, d_info[0])
            d_twin = d.get_beta(2)
            if d_twin is None:
                boundary_darts.append(d)
        return boundary_darts

    def node_in_mesh(self, x: float, y: float) -> (bool, int):
        """
        Search if the node of coordinate (x, y) is inside the mesh.
        :param x: X coordinate
        :param y: Y coordinate
        :return: a boolean indicating if the node is inside the mesh and the id of the node if it is.
        """
        n_id = 0
        for n in self.mesh.nodes:
            if n[2] >= 0:
                if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
                    return True, n_id
            n_id += 1
        return False, None

    def check_beta2_relation(self, dart_info) -> bool:
        """
        Check if beta2 relation is well-defined.
            * beta2(d) == d2 & beta2(d2) == d
            * beta2(d) != beta1(d)
            * if beta2(d) is None, n1 and n2 are boundary nodes
            * The nodes at the ends of d and d2 must be the same.
            * d and d2 must be on different faces
        :param dart_info: dart info to check
        :return: True if all checks passed, raise a Value Error otherwise
        """
        d_id = dart_info[0]
        d1_id = dart_info[1]
        d2_id = dart_info[2]
        n1 = Dart(self.mesh, d_id).get_node()
        na1 = NodeAnalysis(n1)
        n2 = Dart(self.mesh, d1_id).get_node()
        na2 = NodeAnalysis(n2)

        if d2_id >= 0:
            # Get nodes at the ends of d2
            d = Dart(self.mesh, d_id)
            d2 = Dart(self.mesh, d2_id)
            n21 = d2.get_node()
            d21 = d2.get_beta(1)
            n22 = d21.get_node()

            if d.get_face() == d2.get_face():
                return False

        if d2_id >= 0 and self.mesh.dart_info[ d2_id, 0] < 0:
            # raise ValueError("error beta2")
            return False
        elif d2_id >= 0 and self.mesh.dart_info[ d2_id, 2] != d_id:
            # raise ValueError("error beta2")
            return False
        elif d1_id == d2_id:
            # raise ValueError("error same beta2 and beta1")
            return False
        elif d2_id <0:
            if not na1.on_boundary() or not na2.on_boundary():
                # raise ValueError("error not a boundary dart")
                return False
        elif d2_id >=0 and (n21 != n2 or n1 != n22):
            # raise ValueError("different nodes at extremities")
            return False

        return True

    def check_beta1_relation(self, dart_info) -> bool:
        """
        Check if beta1 relation is well-defined.
            * beta1⁴(d) == d
        :param dart_info: dart info to check
        :return: True if all checks passed, raise a Value Error otherwise
        """
        d_id = dart_info[0]
        d1_id = dart_info[1]
        d1 = Dart(self.mesh, d1_id)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)
        if d111.get_beta(1).id != d_id:
            return False
            # raise ValueError("error beta1")
        return True

    def check_faces(self) -> bool:
        """
        Check if all faces are connected to others and if they are composed of 4 different nodes.
        :return: True if all checks passed, false if at least one face is isolated
        """
        for f in self.mesh.active_faces():
            d = Dart(self.mesh, f)
            d1 = d.get_beta(1)
            d11 = d1.get_beta(1)
            d111 = d11.get_beta(1)
            d2 = d.get_beta(2)
            d12 = d1.get_beta(2)
            d112 = d11.get_beta(2)
            d1112 = d111.get_beta(2)
            if d2 is None and d12 is None and d112 is None and d1112 is None:
                return False
            f_nodes_id = [d.get_node().id, d1.get_node().id, d11.get_node().id, d111.get_node().id]
            if len(set(f_nodes_id)) != 4:
                return False
        return True

    def check_double(self, dart_info) -> bool:
        n_from_id = dart_info[3]
        n_to_id = Dart(self.mesh, dart_info[1]).get_node()

        for dart_info_double in self.mesh.active_darts():
            if dart_info_double[0] != dart_info[0]:
                n_from_id_double = dart_info_double[3]
                n_to_id_double = Dart(self.mesh, dart_info_double[1]).get_node()
                if n_from_id_double == n_from_id and n_to_id_double == n_to_id:
                    return False

        return True


    def mesh_check(self) -> bool:
        valid = True
        for dart_info in self.mesh.active_darts():
            valid = self.check_beta2_relation(dart_info) and self.check_beta1_relation(dart_info) # and self.check_double(dart_info)
            if not valid:
                return valid
        # Faces must be composed of 4 different nodes. If there is more than one face, none of them should be isolated.
        if len(self.mesh.active_faces()) != 1:
            valid = self.check_faces()
        return valid