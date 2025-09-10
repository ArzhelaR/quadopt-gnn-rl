from __future__ import annotations
import warnings

from copy import deepcopy

from mesh_model.mesh_analysis.global_mesh_analysis import NodeAnalysis
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Node, Dart
from mesh_model.reader import read_gmsh
from view.mesh_plotter.mesh_plots import plot_mesh

"""
Quadrangular actions performed on meshes.

Each function returns one Boolean:
    * action_validity: If the action has been performed
"""


def flip_edge_cntcw_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return flip_edge_cntcw(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)

def flip_edge_cntcw(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if check_mesh_structure:
        mesh_before = deepcopy(mesh_analysis.mesh)

    if found:
        topo = mesh_analysis.isFlipCCWOk(d)
        if not topo:
            return False
    else:
        return False


    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh_analysis.mesh.active_quadrangles(d)

    f1 = d.get_face()
    f2 = d2.get_face()

    # Update beta 1
    d.set_beta(1, d11)
    d111.set_beta(1, d21)
    d21.set_beta(1, d)
    d2.set_beta(1, d211)
    d2111.set_beta(1, d1)
    d1.set_beta(1, d2)

    #Update nodes links
    if n1.get_dart().id == d.id:
        n1.set_dart(d21)
    if n2.get_dart().id == d2.id:
        n2.set_dart(d1)

    if f1.get_dart().id == d1.id:
        f1.set_dart(d)
    if f2.get_dart().id == d21.id:
        f2.set_dart(d2)

    d.set_node(n5)
    d2.set_node(n3)
    d21.set_face(f1)
    d1.set_face(f2)

    # update nodes scores
    n1.set_score(n1.get_score() + 1)
    n2.set_score(n2.get_score() + 1)
    n3.set_score(n3.get_score() - 1)
    n5.set_score(n5.get_score() - 1)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            raise ValueError("Some checks are missing")

    return True

def flip_edge_cw_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return flip_edge_cw(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)

def flip_edge_cw(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if found:
        topo= mesh_analysis.isFlipCWOk(d)
        if not topo:
            return False
    else:
        return False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh_analysis.mesh.active_quadrangles(d)

    f1 = d.get_face()
    f2 = d2.get_face()

    # Update beta 1
    d.set_beta(1, d2111)
    d2111.set_beta(1, d1)
    d11.set_beta(1, d)

    d111.set_beta(1, d21)
    d211.set_beta(1, d2)
    d2.set_beta(1, d111)

    # update nodes links
    if n1.get_dart().id == d.id:
        n1.set_dart(d21)
    if n2.get_dart().id == d2.id:
        n2.set_dart(d1)

    if f1.get_dart().id == d111.id:
        f1.set_dart(d)

    if f2.get_dart().id == d2111.id:
        f2.set_dart(d2)

    d.set_node(n4)
    d2.set_node(n6)
    d2111.set_face(f1)
    d111.set_face(f2)

    # update nodes scores
    n1.set_score(n1.get_score() + 1)
    n2.set_score(n2.get_score() + 1)
    n4.set_score(n4.get_score() - 1)
    n6.set_score(n6.get_score() - 1)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            raise ValueError("Some checks are missing")

    return True

def split_edge_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return split_edge(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)

def split_edge(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    if check_mesh_structure:
        mesh_before = deepcopy(mesh_analysis.mesh)
    mesh_before = deepcopy(mesh_analysis.mesh)
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if found:
        topo = mesh_analysis.isSplitOk(d)
        if not topo:
            return False
    else:
        return False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh_analysis.mesh.active_quadrangles(d)
    d1112 = d111.get_beta(2)
    d212 = d21.get_beta(2)
    n4_score = n4.get_score()
    # create a new node in the middle of [n1, n2]
    N10 = mesh_analysis.mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)

    # modify existing triangles
    d.set_node(N10)
    d21.set_node(N10)

    # create a new quadrangle
    f5 = mesh_analysis.mesh.add_quad(n1, n5, N10, n4, training=True)

    # update beta2 relations
    mesh_analysis.mesh.set_face_beta2(f5,[d111,d1112,d21,d212])

    # update nodes scores
    n1.set_score(n1.get_score() + 1)
    n4.set_score(n4.get_score() - 1)
    n5.set_score(n5.get_score() - 1)
    N10.set_score(1)  # new nodes have an adjacency of 3, wich means a score of 1
    N10.set_ideal_adjacency(4)  # the inner vertices of quadrangular meshes have an ideal adjacency of 4

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            d_before = Dart(mesh_before, d.id)
            ma_before = QuadMeshTopoAnalysis(mesh_before)
            ma_before.isSplitOk(d_before)
            raise ValueError("Some checks are missing")

    return True


def collapse_edge_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return collapse_edge(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)


def collapse_edge(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    mesh = mesh_analysis.mesh
    if check_mesh_structure:
        mesh_before = deepcopy(mesh)
    found, d = mesh.find_inner_edge(n1, n2)
    if found:
        topo = mesh_analysis.isCollapseOk(d)
        if not topo:
            return False
    else:
        return False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    n1_score = n1.get_score()

    if d.id == 93 and n1_score == -2.0:
        plot_mesh(mesh, debug=True)

    d1112 = d111.get_beta(2)
    d12 = d1.get_beta(2)
    d112 = d11.get_beta(2)

    n3_analysis = NodeAnalysis(n3)
    # Move n3 node in the middle of [n3, n1]
    if not n3_analysis.on_boundary():
        n3.set_xy((n3.x() + n1.x()) / 2, (n1.y() + n3.y()) / 2)

    # Check if nodes n2 and n4 are not linked to deleted dart (n3 will be checked after)
    if n2.get_dart() == d1:
        if mesh.is_dart_active(d2):
            n2.set_dart(d2)
        else:
            n2.set_dart(d12.get_beta(1))
    if n4.get_dart() == d111:
        if mesh.is_dart_active(d112):
            n4.set_dart(d112)
        else:
            n4.set_dart(d1112.get_beta(1))

    # Delete the face F5
    f5 = d.get_face()
    mesh_analysis.mesh.del_quad(d, d1, d11, d111, f5)

    n_from = n1
    n_to = n3

    n_from_analysis = NodeAnalysis(n_from)
    adj_darts = n_from_analysis.adjacent_darts()

    for da in adj_darts:
        if da.get_node() == n_from:
            da.set_node(n_to)

    # Change n3 dart association
    for da in adj_darts:
        if da.get_node() == n_to:
            n_to.set_dart(d)
            break

    mesh_analysis.mesh.del_node(n_from)

    # Update beta2 relations
    if d2 is not None:
        d2.set_beta(2, d12)
        d21 = d2.get_beta(1)
        d21.set_node(n3)
    if d1112 is not None:
        d1112.set_beta(2, d112)
        d1112.set_node(n3)
    if d12 is not None:
        d12.set_beta(2, d2)
    if d112 is not None:
        d112.set_beta(2, d1112)

    # update nodes scores
    n2.set_score(n2.get_score() + 1)
    n4.set_score(n4.get_score() + 1)
    n3.set_score(n3.get_score() + n1_score - 2)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            raise ValueError("Some checks are missing")
    return True


def cleanup_edge_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return cleanup_edge(mesh_analysis.mesh, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)

def cleanup_edge(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    mesh = mesh_analysis.mesh
    if check_mesh_structure:
        mesh_before = deepcopy(mesh)

    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)
    if found:
        d_id = d.id
        topo, delete_n_from = mesh_analysis.isCleanupOk(d) # delete_n_from is True if n_from cord can be deleted, otherwise False
        if not topo:
            return False
    else:
        return False

    # nfrom nodes correspond to all the parallel nodes to be deleted
    # nto nodes correspond to all parallel nodes to be merged

    # The two extremities nodes are not retrieved by my method, so we do it first manually
    parallel_darts = mesh_analysis.mesh.find_parallel_darts(d)
    last_dart = parallel_darts[-1]
    ld1 = last_dart.get_beta(1)
    ld11 = ld1.get_beta(1)
    ld111 = ld11.get_beta(1)
    last_node_from = ld111.get_node()
    last_node_to = ld11.get_node()

    if delete_n_from:
        n_to_delete = last_node_from
        n_to = last_node_to
    else:
        n_to_delete = last_node_to
        n_to = last_node_from
    na_node_to_delete = NodeAnalysis(n_to_delete)
    adj_darts = na_node_to_delete.adjacent_darts()
    for da in adj_darts:
        if da.get_node() == n_to_delete:
            da.set_node(n_to)
    mesh_analysis.mesh.del_node(n_to_delete)

    for dp in parallel_darts:
        f = dp.get_face()
        d1 = dp.get_beta(1)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)

        n_from = dp.get_node()
        n_to = d1.get_node()

        if delete_n_from:
            n_to_delete = n_from
        else:
            n_to_delete = n_to
            n_to = n_from

        n_del_score = n_to_delete.get_score() # for later calculation
        n_to_score = n_to.get_score()

        na_node_to_delete = NodeAnalysis(n_to_delete)
        adj_darts = na_node_to_delete.adjacent_darts()
        for da in adj_darts:
            if da.get_node() == n_to_delete:
                da.set_node(n_to)
        mesh_analysis.mesh.del_node(n_to_delete)

        # update beta 2 relations
        d12 = d1.get_beta(2)
        d1112 = d111.get_beta(2)
        if d1112 is not None:
            d1112.set_beta(2, d12)
        if d12 is not None:
            d12.set_beta(2, d1112)

        mesh_analysis.mesh.del_quad(dp, d1, d11, d111, f)

        #Update score
        na_nto = NodeAnalysis(n_to)
        if na_nto.on_boundary():
            n_to.set_score(n_to_score + n_del_score - 2)
        else:
            n_to.set_score(n_to_score + n_del_score - 4)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            mesh_check = mesh_analysis.mesh_check()
            d_test = Dart(mesh_before, d_id)
            ma_before = QuadMeshTopoAnalysis(mesh_before)
            ma_before.isCleanupOk(d_test)
            raise ValueError("Some checks are missing")
    return True

def cleanup_boundary_edge_ids(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return cleanup_edge(mesh_analysis.mesh, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)

def cleanup_boundary_edge(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    mesh = mesh_analysis.mesh
    if check_mesh_structure:
        mesh_before = deepcopy(mesh)

    found, d = mesh_analysis.mesh.find_boundary_edge(n1, n2)
    if found:
        d_id = d.id
        valid = mesh_analysis.isCleanupBoundaryOk(d) # delete_n_from is True if n_from cord can be deleted, otherwise False
        if not valid:
            return False
    else:
        return False

    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d111 = d11.get_beta(1)
    d12 = d1.get_beta(2)
    d1112 = d111.get_beta(2)

    n1 = d.get_node()
    n2 = d1.get_node()
    n3 = d11.get_node()
    n4 = d111.get_node()

    if d12 is None:
        if n1.get_ideal_adjacency() !=3:
            n_to_move = n1
            n_to_move_to = n2
            n_to_delete = n3
            n_no_change = n4
        else:
            n_to_move = n4
            n_to_move_to = n3
            n_to_delete = n2
            n_no_change = n1
        d2_to = d1112
    elif d1112 is None:
        if n1.get_ideal_adjacency() != 3:
            n_to_move_to = n1
            n_to_move = n2
            n_to_delete = n4
            n_no_change = n3
        else:
            n_to_move_to = n4
            n_to_move = n3
            n_to_delete = n1
            n_no_change = n2
        d2_to = d12
    else:
        raise ValueError("Some checks are missing")

    n_move_analysis = NodeAnalysis(n_to_move)
    adj_darts = n_move_analysis.adjacent_darts()
    n_nc_score = n_no_change.get_score()
    n_move_score = n_to_move.get_score()

    if d2_to.get_node() == n_to_move:
        n_to_move_to.set_dart(d2_to)
    else:
        n_to_move_to.set_dart(d2_to.get_beta(1))

    for da in adj_darts:
        if da.get_node() == n_to_move:
            da.set_node(n_to_move_to)

    d2_to.set_beta(2, None)
    mesh_analysis.mesh.del_node(n_to_move)
    mesh_analysis.mesh.del_node(n_to_delete)
    mesh_analysis.mesh.del_quad(d, d1, d11, d111, d.get_face())

    n_no_change.set_score(n_nc_score - 1)
    n_to_move_to.set_score(n_move_score - 1)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            mesh_check = mesh_analysis.mesh_check()
            d_test = Dart(mesh_before, d_id)
            ma_before = QuadMeshTopoAnalysis(mesh_before)
            ma_before.isCleanupOk(d_test)
            raise ValueError("Some checks are missing")
    return True

def fuse_faces(mesh_analysis, id1: int, id2: int, check_mesh_structure=True) -> True:
    return fuse_faces(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2), check_mesh_structure)


def fuse_faces(mesh_analysis, n1: Node, n2: Node, check_mesh_structure=True) -> True:
    mesh = mesh_analysis.mesh
    if check_mesh_structure:
        mesh_before = deepcopy(mesh)

    found, d = mesh.find_inner_edge(n1, n2)
    if found:
        topo = mesh_analysis.isFuseOk(d)
        if not topo:
            return False
    else:
        return False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)
    #plot_mesh(mesh_before, debug=True)
    f = d.get_face()
    f_to_delete = d2.get_face()

    f.set_dart(d111)

    d211.set_beta(1, d11)
    d111.set_beta(1, d21)
    d21.set_face(f)
    d211.set_face(f)

    if n3.get_dart() == d2111:
        n3.set_dart(d11)
    if n1.get_dart() == d:
        n1.set_dart(d2)

    mesh.del_quad(d,d1,d2,d2111,f_to_delete)

    mesh.del_node(n2)

    if check_mesh_structure:
        mesh_check = mesh_analysis.mesh_check()
        if not mesh_check:
            plot_mesh(mesh_before, debug=True)
            plot_mesh(mesh_analysis.mesh, debug=True)
            raise ValueError("Some checks are missing")
    return True



"""
Old cleanup
"""

# nodes_d = [d.get_node(), d1.get_node()]  # [node_from, node_to]
# nodes_d11p = [d111.get_node(), d11.get_node()]  # [node_from, node_to]
#
# na_df = NodeAnalysis(nodes_d[0])
# na_dt = NodeAnalysis(nodes_d[1])
#
# if nodes_d[0].get_ideal_adjacency() == 3:
#     d_node_to_remove = nodes_d[0]
#     d_node_to = nodes_d[1]
# elif nodes_d[1].get_ideal_adjacency() == 3:
#     d_node_to_remove = nodes_d[1]
#     d_node_to = nodes_d[0]
#
# if nodes_d11p[0].get_ideal_adjacency() == 3:
#     d11p_node_to_remove = nodes_d11p[0]
#     d11p_node_to = nodes_d11p[1]
# elif nodes_d11p[1].get_ideal_adjacency() == 3:
#     d11p_node_to_remove = nodes_d11p[1]
#     d11p_node_to = nodes_d11p[0]
#
# # nfrom nodes correspond to all the parallel nodes to be deleted
# # nto nodes correspond to all parallel nodes to be merged
#
# # The two extremities nodes are not retrieved by my method, so we do it first manually
# parallel_darts = mesh_analysis.mesh.find_parallel_darts(d)
# last_dart = parallel_darts[-1]
# ld1 = last_dart.get_beta(1)
# ld11 = ld1.get_beta(1)
# ld111 = ld11.get_beta(1)
# last_node_from = ld111.get_node()
# last_node_to = ld11.get_node()
#
# naf = NodeAnalysis(last_node_from)
# nat = NodeAnalysis(last_node_to)
# if naf.on_boundary() and last_node_from.get_ideal_adjacency() != 3 and nat.on_boundary() and last_node_to.get_ideal_adjacency() == 3:
#     # We can't delete last node from, but last node to can be deleted
#     na_node_to_delete = nat
#     n_to_delete = last_node_to
#     n_to = last_node_from
# elif naf.on_boundary() and nat.on_boundary() and last_node_from.get_ideal_adjacency() == 3 and last_node_to.get_ideal_adjacency() != 3:
#     na_node_to_delete = naf
#     n_to_delete = last_node_from
#     n_to = last_node_to
# else:
#     raise ValueError("No node to delete found")
# adj_darts = na_node_to_delete.adjacent_darts()
# for da in adj_darts:
#     if da.get_node() == n_to_delete:
#         da.set_node(n_to)
# mesh_analysis.mesh.del_node(n_to_delete)
#
# dp = parallel_darts[-1]
#
# f = dp.get_face()
# d1 = dp.get_beta(1)
# d11 = d1.get_beta(1)
# d111 = d11.get_beta(1)
#
# n_from = dp.get_node()
# n_to = d1.get_node()
#
# naf = NodeAnalysis(n_from)
# nat = NodeAnalysis(n_to)
# if naf.on_boundary() and n_from.get_ideal_adjacency() != 3 and nat.on_boundary() and n_to.get_ideal_adjacency() == 3:
#     # We can't delete last node from, but last node to can be deleted
#     na_node_to_delete = nat
#     n_to_delete = n_to
#     n_to = n_from
# elif naf.on_boundary() and nat.on_boundary() and n_from.get_ideal_adjacency() == 3 and n_to.get_ideal_adjacency() != 3:
#     na_node_to_delete = naf
#     n_to_delete = n_from
#     n_to = n_to
# else:
#     raise ValueError("No node to delete found")
#
# n_del_score = n_to_delete.get_score()  # for later calculation
# n_to_score = n_to.get_score()
#
# adj_darts = na_node_to_delete.adjacent_darts()
# for da in adj_darts:
#     if da.get_node() == n_to_delete:
#         da.set_node(n_to)
#
# mesh_analysis.mesh.del_node(n_to_delete)
#
# # update beta 2 relations
# d12 = d1.get_beta(2)
# d1112 = d111.get_beta(2)
# if d1112 is not None:
#     d1112.set_beta(2, d12)
# if d12 is not None:
#     d12.set_beta(2, d1112)
#
# mesh_analysis.mesh.del_quad(dp, d1, d11, d111, f)
#
# # Update score
# na_nto = NodeAnalysis(n_to)
# if na_nto.on_boundary():
#     n_to.set_score(n_to_score + n_del_score - 2)
# else:
#     n_to.set_score(n_to_score + n_del_score - 4)
