import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import rlcard
from rlcard.utils import get_device

def copy_env(env):
    env_copy = rlcard.make('no-limit-holdem', config={'game_num_players': 3})
    env_copy.game = copy.deepcopy(env.game)
    return env_copy


class DeepCFRNetwork(nn.Module):
    """Neural network for Deep CFR agent"""
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(DeepCFRNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.network(x)

class DeepCFRAgent:
    def __init__(self, num_actions, state_shape, hidden_dim=128, lr=0.001,
                 memory_size=1000000, batch_size=32, device='cpu'):
        self.use_raw = False
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        if isinstance(state_shape, tuple):
            self.input_dim = np.prod(state_shape)
        else:
            self.input_dim = state_shape

        self.regret_network = DeepCFRNetwork(self.input_dim, hidden_dim, num_actions).to(device)
        self.strategy_network = DeepCFRNetwork(self.input_dim, hidden_dim, num_actions).to(device)

        self.regret_optimizer = optim.Adam(self.regret_network.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_network.parameters(), lr=lr)

        self.regret_memory = []
        self.strategy_memory = []
        self.memory_size = memory_size

        self.iteration = 0
        self.training_mode = True

    def _state_to_tensor(self, state):
        if 'obs' in state:
            obs = state['obs']
        else:
            obs = np.zeros(self.input_dim)

        obs = np.array(obs).flatten()
        if len(obs) < self.input_dim:
            obs = np.pad(obs, (0, self.input_dim - len(obs)), 'constant')
        else:
            obs = obs[:self.input_dim]

        return torch.FloatTensor(obs).to(self.device)

    def _get_regret_matching_strategy(self, regrets):
        positive_regrets = torch.clamp(regrets, min=0)
        regret_sum = torch.sum(positive_regrets)
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            return torch.ones_like(regrets) / len(regrets)

    def step(self, state):
        if not state.get('legal_actions'):
            return 0

        legal_actions = list(state['legal_actions'].keys())
        state_tensor = self._state_to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            regrets = self.regret_network(state_tensor).squeeze(0)

        strategy = self._get_regret_matching_strategy(regrets)

        masked_strategy = torch.zeros(self.num_actions)
        for action in legal_actions:
            if action < self.num_actions:
                masked_strategy[action] = strategy[action]

        strategy_sum = torch.sum(masked_strategy)
        if strategy_sum > 0:
            masked_strategy = masked_strategy / strategy_sum
        else:
            for action in legal_actions:
                masked_strategy[action] = 1.0 / len(legal_actions)

        if self.training_mode:
            probs = masked_strategy.numpy()
            probs += np.random.dirichlet([0.1] * self.num_actions) * 0.1
            probs /= np.sum(probs)
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = torch.argmax(masked_strategy).item()

        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        return action

    def eval_step(self, state):
        if not state.get('legal_actions'):
            return 0, {'probs': {}}

        legal_actions = list(state['legal_actions'].keys())
        self.training_mode = False

        state_tensor = self._state_to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            logits = self.strategy_network(state_tensor).squeeze(0)
            strategy = torch.softmax(logits, dim=0)

        masked_strategy = torch.zeros(self.num_actions)
        for action in legal_actions:
            if action < self.num_actions:
                masked_strategy[action] = strategy[action]

        strategy_sum = torch.sum(masked_strategy)
        if strategy_sum > 0:
            masked_strategy /= strategy_sum
        else:
            for action in legal_actions:
                masked_strategy[action] = 1.0 / len(legal_actions)

        action = torch.argmax(masked_strategy).item()
        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        probs = [0.0] * self.num_actions
        for i in range(self.num_actions):
            probs[i] = masked_strategy[i].item()

        info = {'probs': {a: probs[a] for a in legal_actions}}
        return action, info

    def update_regret_memory(self, state, action, regret):
        if len(self.regret_memory) >= self.memory_size:
            self.regret_memory.pop(0)
        self.regret_memory.append((state, action, regret))

    def update_strategy_memory(self, state, strategy):
        if len(self.strategy_memory) >= self.memory_size:
            self.strategy_memory.pop(0)
        self.strategy_memory.append((state, strategy.detach().cpu()))

    def train_networks(self):
        if len(self.regret_memory) < self.batch_size or len(self.strategy_memory) < self.batch_size:
            return
        self._train_regret_network()
        self._train_strategy_network()
        self.iteration += 1

    def _train_regret_network(self):
        batch = random.sample(self.regret_memory, self.batch_size)
        states, actions, regrets = [], [], []
        for s, a, r in batch:
            states.append(self._state_to_tensor(s))
            actions.append(a)
            regrets.append(r)
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        regrets = torch.FloatTensor(regrets).to(self.device)

        predicted = self.regret_network(states)
        loss = nn.MSELoss()(predicted.gather(1, actions.unsqueeze(1)).squeeze(), regrets)

        self.regret_optimizer.zero_grad()
        loss.backward()
        self.regret_optimizer.step()

    def _train_strategy_network(self):
        batch = random.sample(self.strategy_memory, self.batch_size)
        states, strategies = [], []
        for s, strat in batch:
            states.append(self._state_to_tensor(s))
            strategies.append(strat)
        states = torch.stack(states)
        strategies = torch.stack(strategies)
        
        # Fix: Ensure strategies tensor is on the correct device
        strategies = strategies.to(self.device)

        logits = self.strategy_network(states)
        probs = torch.softmax(logits, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(probs + 1e-8), strategies)

        self.strategy_optimizer.zero_grad()
        loss.backward()
        self.strategy_optimizer.step()

    def traverse_tree(self, env, state, player):
        if env.is_over():
            # In RLCard, payoffs are typically obtained from the environment
            # after the game ends, not from the state dictionary
            payoffs = env.get_payoffs()
            return payoffs[player]

        current_player = env.get_player_id()

        if current_player == player:
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            regrets = self.regret_network(state_tensor).detach().squeeze(0)
            strategy = self._get_regret_matching_strategy(regrets)

            # Fix: Ensure action_utils is on the same device as strategy
            action_utils = torch.zeros(self.num_actions, device=self.device)

            legal_actions = list(state['legal_actions'].keys())

            for action in legal_actions:
                env_copy = copy_env(env)
                # Fix: RLCard step() returns (next_state, next_player_id)
                next_state, next_player_id = env_copy.step(action)
                utility = self.traverse_tree(env_copy, next_state, player)
                # Convert utility to tensor on the correct device
                action_utils[action] = torch.tensor(utility, device=self.device)

            node_util = torch.dot(strategy, action_utils)
            advantages = action_utils - node_util

            for action in legal_actions:
                self.update_regret_memory(state, action, advantages[action].item())

            self.update_strategy_memory(state, strategy)

            return node_util.item()
        
        else:
            legal_actions = list(state['legal_actions'].keys())
            action = np.random.choice(legal_actions)
            # Fix: RLCard step() returns (next_state, next_player_id)
            next_state, next_player_id = env.step(action)
            return self.traverse_tree(env, next_state, player)
        
    def sample_advantage_memory(self, env, num_traversals=100):
        for _ in range(num_traversals):
            state, player_id = env.reset()
            self.traverse_tree(env, state, player=player_id)

    def save_model(self, path):
        torch.save({
            'regret_network': self.regret_network.state_dict(),
            'strategy_network': self.strategy_network.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'iteration': self.iteration
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.regret_network.load_state_dict(checkpoint['regret_network'])
        self.strategy_network.load_state_dict(checkpoint['strategy_network'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        self.iteration = checkpoint['iteration']
