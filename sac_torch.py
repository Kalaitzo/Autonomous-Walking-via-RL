import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, gamma=0.99, input_dims=[8], env=None, n_actions=2, max_size=1000000,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)  # Replay buffer

        # Agent networks
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name='actor', max_actions=env.action_space.high)  # Actor network

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')  # Critic network 1
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')  # Critic network 2

        self.value = ValueNetwork(beta, input_dims, name='value')  # Value network
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')  # Target value network

        self.scale = reward_scale  # Reward scale
        self.update_network_parameters(tau=1)  # Update target value network

    def choose_action(self, observation):  # Choose action
        state = T.Tensor([observation]).to(self.actor.device)  # Convert observation to tensor
        actions, _ = self.actor.sample_normal(state, reparameterize=False)  # Sample action

        return actions.cpu().detach().numpy()[0]  # Return action

    def remember(self, state, action, reward, new_state, done):  # Remember
        self.memory.store_transition(state, action, reward, new_state, done)  # Store transition

    def update_network_parameters(self, tau=None):  # Update network parameters
        if tau is None:  # If tau is none
            tau = self.tau

        # Update target value network
        target_value_params = self.target_value.named_parameters()  # Target value network parameters
        value_params = self.value.named_parameters()  # Value network parameters

        target_value_state_dict = dict(target_value_params)  # Target value network state dictionary
        value_state_dict = dict(value_params)  # Value network state dictionary

        for name in value_state_dict:  # Update target value network parameters
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)  # Load target value network state dictionary

    def save_models(self):  # Save models
        print('... saving models ...')
        self.actor.save_checkpoint()  # Save actor network
        self.critic_1.save_checkpoint()  # Save critic network 1
        self.critic_2.save_checkpoint()  # Save critic network 2
        self.value.save_checkpoint()  # Save value network
        self.target_value.save_checkpoint()  # Save target value network

    def load_models(self):  # Load models
        print('... loading models ...')
        self.actor.load_checkpoint()  # Load actor network
        self.critic_1.load_checkpoint()  # Load critic network 1
        self.critic_2.load_checkpoint()  # Load critic network 2
        self.value.load_checkpoint()  # Load value network
        self.target_value.load_checkpoint()  # Load target value network

    def learn(self):  # Learn
        if self.memory.mem_cntr < self.batch_size:  # If the memory counter is less than batch size
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)  # Sample buffer

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)  # Convert reward to tensor
        done = T.tensor(done).to(self.actor.device)  # Convert done to tensor
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)  # Convert new state to tensor
        state = T.tensor(state, dtype=T.float).to(self.actor.device)  # Convert state to tensor
        action = T.tensor(action, dtype=T.float).to(self.actor.device)  # Convert action to tensor

        value = self.value(state).view(-1)  # Calculate V(s) from with the value network
        value_ = self.target_value(state_).view(-1)  # Calculate V(s) hat from with the target value network
        value_[done] = 0.0  # Target value

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)  # Sample action
        log_probs = log_probs.view(-1)  # Log probabilities
        q1_new_policy = self.critic_1.forward(state, actions)  # Critic network 1
        q2_new_policy = self.critic_2.forward(state, actions)  # Critic network 2
        critic_value = T.min(q1_new_policy, q2_new_policy)  # Critic value
        critic_value = critic_value.view(-1)  # Critic value

        # Calculate value loss
        self.value.optimizer.zero_grad()  # Zero gradients
        value_target = critic_value - log_probs  # Value target
        value_loss = 0.5 * F.mse_loss(value, value_target)  # Value loss
        value_loss.backward(retain_graph=True)  # Backward propagation
        self.value.optimizer.step()  # Step

        # Calculate policy loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)  # Sample action
        log_probs = log_probs.view(-1)  # Log probabilities
        q1_new_policy = self.critic_1.forward(state, actions)  # Critic network 1
        q2_new_policy = self.critic_2.forward(state, actions)  # Critic network 2
        critic_value = T.min(q1_new_policy, q2_new_policy)  # Critic value
        critic_value = critic_value.view(-1)  # Critic value

        actor_loss = log_probs - critic_value  # Actor loss
        actor_loss = T.mean(actor_loss)  # Actor loss
        self.actor.optimizer.zero_grad()  # Zero gradients
        actor_loss.backward(retain_graph=True)  # Backward propagation
        self.actor.optimizer.step()  # Step

        # Calculate critic loss
        self.critic_1.optimizer.zero_grad()  # Zero gradients
        self.critic_2.optimizer.zero_grad()  # Zero gradients
        q_hat = self.scale * reward + self.gamma * value_  # Q hat
        q1_old_policy = self.critic_1.forward(state, action).view(-1)  # Critic network 1
        q2_old_policy = self.critic_2.forward(state, action).view(-1)  # Critic network 2
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)  # Critic loss 1
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)  # Critic loss 2

        critic_loss = critic_1_loss + critic_2_loss  # Critic loss
        critic_loss.backward()  # Backward propagation
        self.critic_1.optimizer.step()  # Step
        self.critic_2.optimizer.step()  # Step

        self.update_network_parameters()  # Update network parameters
