import copy
import random
from copy import deepcopy

import pygame
import imageio
import sys
import os

import numpy as np
import gymnasium as gym

from enum import Enum
from typing import Optional
from pygame.locals import *

from mesh_model.random_quadmesh import random_mesh
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshTopoAnalysis
from environment.quadmesh_env.envs.mesh_conv import get_x
from environment.actions.quadrangular_actions import flip_edge_cntcw, flip_edge_cw, split_edge, collapse_edge, \
    cleanup_edge, cleanup_boundary_edge, fuse_faces
from environment.observation_register import ObservationRegistry
from view.window import window_data, graph
from mesh_display import MeshDisplay


class Actions(Enum):
    FLIP_CW = 0
    FLIP_CNTCW = 1
    SPLIT = 2
    COLLAPSE = 3
    CLEANUP_BDY = 4
    FUSE = 5
    CLEANUP = 6



class QuadMeshEnv(gym.Env):
    """
    QuadMesh environment is structured according to gymnasium and is used to topologically optimize quadrangular meshes topologically.
    The generated observations consist of a local topological view of the mesh. They are structured in the form of matrices :
        * The columns correspond to the surrounding area of a mesh dart.
        * Only the darts with the most irregularities in the surrounding area are retained.

    Based on these observations, the agent will choose from 4 different actions:
        * flip clockwise, flip an edge clockwise
        * flip counterclockwise, flip an edge counterclockwise
        * split, add a face
        * collapse, deleting a face

    These actions will generate rewards proportional to the improvement or deterioration of the mesh. If the chosen action is invalid, a penalty is returned.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            learning_mesh=None,
            max_episode_steps: int = 50,
            n_darts_selected: int = 20,
            deep: int = 6,
            render_mode:  Optional[str] = None,
            with_degree_obs: bool = True,
            action_restriction: bool = False,
            obs_count: bool = False,
            analysis_type = "topo",
            debug = True
    ) -> None:

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If a mesh has been entered, it is used, else if it's a dataset, a random mesh is picked from it, otherwise a random mesh is generated.
        if isinstance(learning_mesh, Mesh):
            self.config = {"mesh": learning_mesh, "learning_meshes": None}
            self.mesh = copy.deepcopy(learning_mesh)
            self.mesh_size = 0
        elif learning_mesh is None:
            self.config = {"mesh": None, "learning_meshes": None}
            self.mesh_size = learning_mesh
            self.mesh = random_mesh()
        elif isinstance(learning_mesh, list):
            self.config = {"mesh": None, "learning_meshes": learning_mesh}
            self.mesh = copy.deepcopy(random.choice(learning_mesh))
            self.mesh_size = 0

        self.n_darts_selected = n_darts_selected
        self.deep = deep
        self.analysis_type = analysis_type
        self.mesh_analysis = QuadMeshTopoAnalysis(self.mesh)
        self.debug = debug
        obs_shape = (self.n_darts_selected, self.deep)

        #self.mesh_size = len(self.mesh.nodes)
        #self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score = self.mesh_analysis.global_score()
        self.last_nodes_scores = None
        self._ideal_rewards = (self._mesh_score - self._ideal_score)*10 #arbitrary factor of 10 for rewards
        self.next_mesh_score = 0
        self.restricted = action_restriction
        self.degree_observation = with_degree_obs
        self.nb_invalid_actions = 0
        self.max_steps = max_episode_steps
        self.episode_count = 0
        self.ep_len = 0
        self.darts_selected = [] # darts id observed

        self.actions_info = {
            "n_flip_cntcw": 0,
            "n_flip_ccw": 0,
            "n_split": 0,
            "n_collapse": 0,
            "n_cleanup_bdy": 0,
            "n_fuse": 0,
            "n_cleanup": 0,
        }

        # Definition of an observation register if required
        if obs_count:
            self.observation_count = True
            self.observation_registry = ObservationRegistry(self.n_darts_selected, self.deep, -6, 2)
        else:
            self.observation_count = False

        # Render
        if self.render_mode == "human":
            self.mesh_disp = MeshDisplay(self.mesh)
            self.graph = graph.Graph(self.mesh_disp.get_nodes_coordinates(), self.mesh_disp.get_edges(),
                                         self.mesh_disp.get_scores())
            self.win_data = window_data()
            self.window_size = 512  # The size of the PyGame window
            self.window = None
            self.clock = None

            self.recording = False
            self.frames = []

            os.makedirs("episode_recordings", exist_ok=True)

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-8,  # nodes min degree : -6
            high=4,  # nodes max degree : 2
            shape=obs_shape,
            dtype=np.int64
        )
        self.observation = None

        # We have 6 actions, flip clockwise, flip counterclockwise, split, collapse, fuse, cleanup_boundary
        self.action_space = gym.spaces.MultiDiscrete([6, self.n_darts_selected])

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if options is not None:
            self.mesh = options['mesh']
        elif self.config["mesh"] is not None:
            self.mesh = copy.deepcopy(self.config["mesh"])
        elif self.config["learning_meshes"] is not None:
            self.mesh = copy.deepcopy(random.choice(self.config["learning_meshes"]))
        else:
            self.mesh = random_mesh()

        self.mesh_analysis = QuadMeshTopoAnalysis(self.mesh)

        self._nodes_scores, self._mesh_score, self._ideal_score = self.mesh_analysis.global_score()
        self.last_nodes_scores = None
        self._ideal_rewards = (self._mesh_score - self._ideal_score) * 10
        self.nb_invalid_actions = 0
        self.close()
        self.observation = self._get_obs()
        self.ep_len = 0
        info = self._get_info(terminated=False,valid_action=None, action=(None,None), mesh_reward=None)
        self.actions_info = {
            "n_flip_cw": 0,
            "n_flip_cntcw": 0,
            "n_split": 0,
            "n_collapse": 0,
            "n_cleanup_bdy": 0,
            "n_fuse": 0,
        }

        if self.render_mode=="human":
            self._render_frame()
            self.recording = True
        else:
            self.recording = False
            self.frames = []

        return self.observation, info

    def _get_obs(self):
        irregularities, darts_list = get_x(self.mesh_analysis, self.n_darts_selected, self.deep, self.analysis_type, self.restricted, self._nodes_scores)
        self.darts_selected = darts_list
        return irregularities

    def _get_info(self, terminated, valid_action, action, mesh_reward):
        return {
            "distance": self._mesh_score - self._ideal_score,
            "mesh_reward" : mesh_reward,
            "mesh_score" : self._mesh_score,
            "mesh_ideal_score" : self._ideal_score,
            "mesh_ideal_rewards" : self._ideal_rewards,
            "is_success": 1.0 if terminated else 0.0,
            "valid_action": 1.0 if valid_action else 0.0,
            "flip_cw": 1.0 if action[0]==Actions.FLIP_CW.value else 0.0,
            "flip_cntcw": 1.0 if action[0]==Actions.FLIP_CNTCW.value else 0.0,
            "split": 1.0 if action[0]==Actions.SPLIT.value else 0.0,
            "collapse": 1.0 if action[0]==Actions.COLLAPSE.value else 0.0,
            "cleanup_bdy": 1.0 if action[0]==Actions.CLEANUP_BDY.value else 0.0,
            "fuse": 1.0 if action[0]==Actions.FUSE.value else 0.0,
            "invalid_flip": 1.0 if (action[0]==Actions.FLIP_CW.value or action[0]==Actions.FLIP_CNTCW.value) and not valid_action else 0.0,
            "invalid_split": 1.0 if action[0]==Actions.SPLIT.value and not valid_action else 0.0,
            "invalid_collapse": 1.0 if action[0]==Actions.COLLAPSE.value and not valid_action else 0.0,
            "invalid_cleanup_bdy": 1.0 if action[0]==Actions.CLEANUP_BDY.value and not valid_action else 0.0,
            "invalid_fuse": 1.0 if action[0]==Actions.FUSE.value and not valid_action else 0.0,
            "mesh" : self.mesh,
            "mesh_analysis" : self.mesh_analysis,
            "darts_selected" : self.darts_selected,
            "observation_registry" : self.observation_registry if self.observation_count else None,
        }

    def _action_to_dart_id(self, action: np.ndarray) -> int:
        """
        Converts an action ID into the dart ID on which to perform the action
        :param action: action ID
        :return: the dart ID on which to perform the action
        """
        return self.darts_selected[int(action[1])]

    def step(self, action: np.ndarray):
        mesh_before = deepcopy(self.mesh)
        self.ep_len+=1
        truncated = False

        if self.observation_count:
            self.observation_registry.register_observation(self.observation)

        dart_id = self._action_to_dart_id(action)
        valid_action = False
        mesh_reward = 0
        reward = 0
        terminated = False

        if dart_id != -1 :
            d = Dart(self.mesh, dart_id)
            d1 = d.get_beta(1)
            n1 = d.get_node()
            n2 = d1.get_node()
            if action[0] == Actions.FLIP_CW.value:
                self.actions_info["n_flip_cw"] += 1
                valid_action = flip_edge_cw(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            elif action[0] == Actions.FLIP_CNTCW.value:
                self.actions_info["n_flip_cntcw"] += 1
                valid_action= flip_edge_cntcw(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            elif action[0] == Actions.SPLIT.value:
                self.actions_info["n_split"] += 1
                valid_action= split_edge(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            elif action[0] == Actions.COLLAPSE.value:
                self.actions_info["n_collapse"] += 1
                valid_action = collapse_edge(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            elif action[0] == Actions.CLEANUP_BDY.value:
                self.actions_info["n_cleanup_bdy"] += 1
                valid_action = cleanup_boundary_edge(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            elif action[0] == Actions.FUSE.value:
                self.actions_info["n_fuse"] += 1
                valid_action = fuse_faces(self.mesh_analysis, n1, n2, check_mesh_structure=self.debug)
            else:
                raise ValueError("Action not defined")

            if valid_action:
                # An episode is done if the actual score is the same as the ideal
                next_nodes_score, self.next_mesh_score, _= self.mesh_analysis.global_score(mesh_before = mesh_before)
                self.last_nodes_scores = self._nodes_scores
                terminated = np.array_equal(self._ideal_score, self.next_mesh_score)
                if terminated:
                    mesh_reward = (self._mesh_score - self.next_mesh_score)*10
                    reward = mesh_reward
                else:
                    mesh_reward = (self._mesh_score - self.next_mesh_score)*10
                    reward = mesh_reward
                self._nodes_scores, self._mesh_score = next_nodes_score, self.next_mesh_score
                self.observation = self._get_obs()
                self.nb_invalid_actions = 0

            elif not valid_action:
                reward = -5
                mesh_reward = 0
                terminated = False
                self.nb_invalid_actions += 1
                if self.nb_invalid_actions > 10:
                    truncated = self.mesh_analysis.isTruncated(self.darts_selected)
            else:
                raise ValueError("Invalid action")

        info = self._get_info(terminated, valid_action, action, mesh_reward)

        if self.render_mode == "human":
            self._render_frame()
        if terminated or self.ep_len>= 200:
            if self.recording and self.frames:
                imageio.mimsave(f"episode_recordings/episode_{self.episode_count}.gif", self.frames, fps=1)
                print("Image recorded")
                self.episode_count +=1

        return self.observation, reward, terminated, truncated, info

    def _render_frame(self):
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.win_data.size, self.win_data.options)
            pygame.display.set_caption('QuadMesh')
            self.window.fill((255, 255, 255))
            self.font = pygame.font.SysFont(None, self.win_data.font_size)
            self.clock = pygame.time.Clock()
            self.clock.tick(60)
            self.win_data.scene_xmin, self.win_data.scene_ymin, self.win_data.scene_xmax, self.win_data.scene_ymax = self.graph.bounding_box()
            self.win_data.scene_center = pygame.math.Vector2((self.win_data.scene_xmax + self.win_data.scene_xmin) / 2.0,
                                                      (self.win_data.scene_ymax + self.win_data.scene_ymin) / 2.0)

        pygame.event.pump()
        self.window.fill((255, 255, 255))  # white
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == VIDEORESIZE or event.type == VIDEOEXPOSE:  # handles window minimising/maximising
                x, y = self.window.get_size()
                text_margin = 200
                self.win_data.center.x = (x - text_margin) / 2
                self.win_data.center.y = y / 2
                ratio = float(x - text_margin) / float(self.win_data.scene_xmax - self.win_data.scene_xmin)
                ratio_y = float(y) / float(self.win_data.scene_ymax - self.win_data.scene_ymin)
                if ratio_y < ratio:
                    ratio = ratio_y

                self.win_data.node_size = max(ratio / 100, 10)
                self.win_data.stretch = 0.75 * ratio

                self.window.fill((255, 255, 255))
                pygame.display.flip()

        self.graph.clear()
        self.mesh_disp = MeshDisplay(self.mesh)
        self.graph.update(self.mesh_disp.get_nodes_coordinates(), self.mesh_disp.get_edges(),
                          self.mesh_disp.get_scores())

        #Draw mesh
        for e in self.graph.edges:
            e.draw(self.window, self.win_data)
        for n in self.graph.vertices:
            n.draw(self.window, self.font, self.win_data)

        #Print action type
        if hasattr(self, 'actions_info'):
            x = self.window.get_width() - 150
            y_start = 100
            line_spacing = 25

            for i, (action_name, count) in enumerate(self.actions_info.items()):
                text = f"{action_name}: {count}"
                text_surface = self.font.render(text, True, (0, 0, 0))
                self.window.blit(text_surface, (x, y_start + i * line_spacing))

        self.clock.tick(60)
        pygame.time.delay(600)
        pygame.display.flip()
        if self.recording:
            pixels = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = pixels.transpose([1,0,2])
            self.frames.append(frame)

    def close(self):
        if self.render_mode=="human" and self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
