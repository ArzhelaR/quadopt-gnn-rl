from gymnasium.envs.registration import register
from environment.quadmesh_env.envs.quadmesh import QuadMeshEnv

register(
    id="Quadmesh-v0",
    entry_point="environment.quadmesh_env.envs:QuadMeshEnv",
    max_episode_steps=150,
    kwargs={"learning_mesh": None, "n_darts_selected": 20, "deep": 6, "with_degree_obs": False, "action_restriction": False},
)
