import gymnasium as gym

from mesh_model.mesh_struct.mesh_elements import Node
from mesh_model.mesh_analysis.global_mesh_analysis import NodeAnalysis


class WeightedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mesh_weighted_score= self.get_weighted_mesh_score()

    def get_weighted_mesh_score(self):
        """
                Calculate the overall weighted mesh score. The mesh cannot achieve a better score than the ideal one.
                And the current score is the mesh score.
                :return: the current mesh weighted score
                """
        mesh_ideal_score = 0
        mesh_score = 0
        nodes_score = []
        nodes_adjacency = []

        mesh = self.get_wrapper_attr('mesh')

        for i in range(len(mesh.nodes)):
            if mesh.nodes[i, 2] >= 0:
                n_id = i
                node = Node(mesh, n_id)
                n_a = NodeAnalysis(node)
                n_score = node.get_score()
                nodes_score.append(n_score)
                nodes_adjacency.append(n_a.degree())
                mesh_ideal_score += n_score
                mesh_score += (n_score)**2
            else:
                nodes_score.append(0)
                nodes_adjacency.append(4)
        return mesh_score

    def step(self, action):
        previous_weighted_mesh_score = self.mesh_weighted_score
        obs, _, terminated, truncated, info = self.env.step(action)
        if info["valid_action"]:
            self.mesh_weighted_score = self.get_weighted_mesh_score()
            reward = (previous_weighted_mesh_score - self.mesh_weighted_score)*10
        else:
            reward = -5
        return obs, reward, terminated, truncated, info
