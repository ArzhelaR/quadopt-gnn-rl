import copy
import random
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical


class NaNExceptionActor(Exception):
    pass


class NaNExceptionCritic(Exception):
    pass


class Actor(nn.Module):
    def __init__(self, env, input_dim, n_actions, n_darts_observed, lr=0.0001, eps=0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions*n_darts_observed)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.9
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.env = env
        self.eps = eps
        self.n_actions = n_actions

    def reset(self, env=None):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.optimizer = Adam(self.parameters(), lr=self.optimizer.defaults['lr'], weight_decay=self.optimizer.defaults['weight_decay'])

    def select_action(self, observation, info):
        ma = info["mesh_analysis"]
        if np.random.rand() < self.eps:
            action = self.env.sample() # random choice of an action
            dart_id = self.env.darts_selected[action[1]]
            action_type = action[0]
            total_actions_possible = np.prod(self.env.action_space.nvec)
            prob = 1/total_actions_possible
        else:
            obs = torch.tensor(observation.flatten(), dtype=torch.float32)
            pmf = self.forward(obs)
            dist = Categorical(pmf)
            action = dist.sample()
            action = action.tolist()
            prob = pmf[action]
            action_dart = int(action/self.n_actions)
            action_type = action % self.n_actions
            dart_id = info["darts_selected"][action_dart]
        action_list = [action, dart_id, action_type]
        return action_list, prob

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise NaNExceptionActor("Les couches cachées renvoient nan ou infinies")
        return self.softmax(x)

    # def get_pi(self, observation):
    #     obs = torch.tensor(observation, dtype=torch.float32)
    #     pmf = self.forward(obs)
    #     return pmf.tolist()
    #
    # def update(self, delta, L, state, action):
    #     obs = self.env._get_obs()
    #     X = torch.tensor(obs, dtype=torch.float32)
    #     action = torch.tensor(action[0], dtype=torch.int64)
    #     pmf = self.forward(X)
    #     log_prob = torch.log(pmf[action])
    #     actor_loss = -log_prob * delta * L
    #     return actor_loss
    #
    # def learn(self, actor_loss ):
    #     self.optimizer.zero_grad()
    #     actor_loss = torch.stack(actor_loss).sum()
    #     actor_loss.backward()
    #     self.optimizer.step()


class PPO:
    def __init__(self, env, obs_size, n_actions, n_darts_observed, max_steps, lr, gamma, nb_iterations, nb_episodes_per_iteration, nb_epochs, batch_size):
        self.env = env
        self.max_steps = max_steps
        self.n_actions =n_actions
        self.actor = Actor(self.env, obs_size, n_actions, n_darts_observed, lr=lr)
        self.lr = lr
        self.gamma = gamma
        self.nb_iterations = nb_iterations
        self.nb_episodes_per_iteration = nb_episodes_per_iteration
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.epsilon = 0.2


    def train(self, dataset):
        num_samples = len(dataset)
        print('training on {}'.format(num_samples))
        for _ in range(self.nb_epochs):
            start = 0
            dataset_rd = random.sample(dataset, num_samples)
            while start < num_samples - 2:
                stop = min(num_samples, start + self.batch_size)
                batch = dataset_rd[start:stop]
                actor_loss = []
                for _, (ma, o, a, r, G, old_prob, next_o, done, st, ideal_s) in enumerate(batch, 1):
                    o = torch.tensor(o.flatten(), dtype=torch.float32)
                    next_o = torch.tensor(next_o.flatten(), dtype=torch.float32)
                    pmf = self.actor.forward(o)
                    log_prob = torch.log(pmf[a[0]])
                    if st == ideal_s:
                        continue
                    advantage = 1 if done else G / 10*(st - ideal_s) # G is the sum of rewards which are multiplicated by 10
                    ratio = torch.exp(log_prob - torch.log(old_prob).detach())
                    actor_loss1 = advantage * ratio
                    actor_loss2 = advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    clipped_obj = min(actor_loss1, actor_loss2)
                    actor_loss.append(-clipped_obj)
                actor_loss = torch.stack(actor_loss).sum()
                self.actor.optimizer.zero_grad()
                with torch.autograd.set_grad_enabled(True):
                    actor_loss.backward()
                self.actor.optimizer.step()
                start = stop + 1

    def learn(self, writer):
        """
        Train the PPO mesh_model
        :return: the actor policy, training rewards, training wins, len of episodes
        """
        rewards = []
        wins = []
        len_ep = []
        valid_actions = []
        global_step = 0
        nb_episodes = 0

        try:
            for iteration in tqdm(range(self.nb_iterations)):
                print('ITERATION', iteration)
                rollouts = []
                dataset = []
                for _ in tqdm(range(self.nb_episodes_per_iteration)):
                    next_obs, info = self.env.reset()
                    trajectory = []
                    ep_reward = 0
                    ep_mesh_reward = 0
                    ep_valid_actions = 0
                    ideal_reward = info["mesh_ideal_rewards"]
                    done = False
                    step = 0
                    while step < self.max_steps:
                        ma = copy.deepcopy(info["mesh_analysis"])
                        obs = next_obs
                        action, prob = self.actor.select_action(obs, info)
                        if action is None:
                            wins.append(0)
                            break
                        gym_action = [action[2],int(action[0]/self.n_actions)]
                        t0 = time.time()
                        next_obs, reward, terminated, truncated, info = self.env.step(gym_action)
                        t = time.time() - t0
                        ep_reward += reward
                        ep_mesh_reward += info["mesh_reward"]
                        ep_valid_actions += info["valid_action"]
                        st = info["mesh_score"]
                        s_ideal = info["mesh_ideal_score"]
                        if terminated:
                            if truncated:
                                wins.append(0)
                                trajectory.append((ma, obs, action, reward, prob, next_obs, done, st, s_ideal))
                            else:
                                wins.append(1)
                                done = True
                                trajectory.append((ma, obs, action, reward, prob, next_obs, done, st, s_ideal))
                            break
                        trajectory.append((ma, obs, action, reward, prob, next_obs, done, st, s_ideal))
                        step += 1
                    if len(trajectory) != 0:
                        #Compute G
                        G = 0
                        computed_trajectory = []
                        for ma, obs, action, reward, prob, next_obs, done, st, s_ideal in reversed(trajectory):
                            G = reward+0.9*G*(1-done)
                            computed_trajectory.append((ma, obs, action, reward, G, prob, next_obs, done, st, s_ideal))
                        computed_trajectory.reverse()
                        rewards.append(ep_reward)
                        valid_actions.append(ep_valid_actions)
                        rollouts.append(computed_trajectory)
                        dataset.extend(computed_trajectory)
                        len_ep.append(len(trajectory))
                    nb_episodes += 1
                    writer.add_scalar("episode_reward", ep_reward, nb_episodes)
                    writer.add_scalar("episode_mesh_reward", ep_mesh_reward, nb_episodes)
                    if ideal_reward !=0 :
                        writer.add_scalar("normalized return", (ep_mesh_reward/ideal_reward), nb_episodes)
                    else :
                        writer.add_scalar("normalized return", ep_mesh_reward, nb_episodes)
                    if len(trajectory) != 0:
                        writer.add_scalar("len_episodes", len(trajectory), nb_episodes)
                        writer.add_scalar("valid_actions", ep_valid_actions*100/len(trajectory), nb_episodes)

                self.train(dataset)

        except NaNExceptionActor:
            print("NaN Exception on Actor Network")
            return None, None, None, None

        return self.actor, rewards, wins, len_ep, None
