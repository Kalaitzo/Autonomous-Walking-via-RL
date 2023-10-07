import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name='critic', fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)  # First layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Second layer
        self.q = nn.Linear(self.fc2_dims, 1)  # Output layer

        self.optimizer = optim.Adam(self.parameters(), lr=beta)  # Adam optimizer
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # GPU or CPU

        self.to(self.device)  # Send to GPU or CPU

    def forward(self, state, action):  # Forward propagation
        action_value = self.fc1(T.cat([state, action], dim=1))  # Concatenate state and action
        action_value = F.relu(action_value)  # ReLU activation
        action_value = self.fc2(action_value)  # Second layer
        action_value = F.relu(action_value)  # ReLU activation

        q = self.q(action_value)  # Output layer

        return q

    def save_checkpoint(self):  # Save checkpoint
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):  # Load checkpoint
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, name='value', fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # First layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Second layer
        self.v = nn.Linear(self.fc2_dims, 1)  # Output layer

        self.optimizer = optim.Adam(self.parameters(), lr=beta)  # Adam optimizer
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # GPU or CPU

        self.to(self.device)  # Send to GPU or CPU

    def forward(self, state):  # Forward propagation
        state_value = self.fc1(state)  # First layer
        state_value = F.relu(state_value)  # ReLU activation
        state_value = self.fc2(state_value)  # Second layer
        state_value = F.relu(state_value)  # ReLU activation

        v = self.v(state_value)  # Output layer

        return v

    def save_checkpoint(self):  # Save checkpoint
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):  # Load checkpoint
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_actions, name='actor',
                 fc1_dims=256, fc2_dims=256, n_actions=2, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_actions = max_actions
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # First layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Second layer
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  # Mean of the distribution for the policy
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)  # Standard deviation of the distribution for the policy

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)  # Adam optimizer
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # GPU or CPU

        self.to(self.device)  # Send to GPU or CPU

    def forward(self, state):  # Forward propagation
        prob = self.fc1(state)  # First layer
        prob = F.relu(prob)  # ReLU activation
        prob = self.fc2(prob)  # Second layer
        prob = F.relu(prob)  # ReLU activation

        mu = self.mu(prob)  # Mean of the distribution for the policy
        sigma = self.sigma(prob)  # Standard deviation of the distribution for the policy

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)  # Clamping

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):  # Sample from normal distribution
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)  # Normal distribution

        if reparameterize:
            actions = probabilities.rsample()  # Add noise to the actions
        else:
            actions = probabilities.sample()  # Sample from the distribution

        action = T.tanh(actions) * T.tensor(self.max_actions).to(self.device)  # Tanh activation
        log_probs = probabilities.log_prob(actions)  # Log probabilities to calculate the loss
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)  # Sum the log probabilities (scalar)

        return action, log_probs

    def save_checkpoint(self):  # Save checkpoint
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):  # Load checkpoint
        self.load_state_dict(T.load(self.checkpoint_file))
