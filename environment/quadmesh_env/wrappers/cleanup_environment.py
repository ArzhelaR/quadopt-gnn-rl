import gymnasium as gym
import numpy as np

from environment.actions.quadrangular_actions import cleanup_edge
from mesh_model.mesh_struct.mesh_elements import Node, Dart


class CleanupWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cpt = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.cpt == 5:
            self.cpt = 0
            m_a = self.get_wrapper_attr('mesh_analysis')
            darts_selected = list(self.get_wrapper_attr('darts_selected'))

            if len(m_a.mesh.active_darts()) > len(darts_selected)*2:
                cleanup_performed = False
                while not cleanup_performed and len(darts_selected) > 0:
                    dart_id = darts_selected.pop(0)
                    if m_a.mesh.dart_info[dart_id, 0] >= 0:
                        d = Dart(m_a.mesh, dart_id)
                        d1 = d.get_beta(1)
                        n1 = d.get_node()
                        n2 = d1.get_node()
                        cleanup_performed = cleanup_edge(m_a, n1, n2)
                self.set_wrapper_attr('mesh_analysis', m_a)
                self.set_wrapper_attr('mesh', m_a.mesh)

                next_nodes_score, next_mesh_score, mesh_ideal_score = m_a.global_score()
                terminated = np.array_equal(mesh_ideal_score, next_mesh_score)
                self.set_wrapper_attr('_nodes_scores', next_nodes_score)
                self.set_wrapper_attr('_mesh_score', next_mesh_score)
                unwrapped_env = self.unwrapped
                obs = unwrapped_env._get_obs()
        else:
            self.cpt += 1
        return obs, reward, terminated, truncated, info
