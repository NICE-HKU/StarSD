import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import nakagami, linregress

    

class Actor_1d(nn.Module):
    """
    Continuous action Actor for SAC using tanh-Gaussian policy.
    Outputs actions in [0, 1] range using tanh squashing.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor_1d, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        
        # Output mean and log_std for Gaussian distribution
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Numerical stability bounds
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        
    def forward(self, state):
        """
        Forward pass to compute mean and log_std
        
        Args:
            state: [batch_size, state_dim]
        Returns:
            mean: [batch_size, action_dim]
            log_std: [batch_size, action_dim]
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Clamp for numerical stability (gradients still flow)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action using reparameterization trick with tanh squashing
        
        Args:
            state: [batch_size, state_dim]
        Returns:
            action: [batch_size, action_dim] in range [0, 1]
            log_prob: [batch_size, 1]
        """
        # Input validation
        if torch.isnan(state).any():
            print("⚠️ Warning: Input state contains NaN! Using zeros.")
            state = torch.zeros_like(state)
        
        mean, log_std = self.forward(state)
        
        # Output validation
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("⚠️ Warning: Actor output abnormal! Using safe defaults.")
            mean = torch.zeros_like(mean)
            log_std = torch.full_like(log_std, -1.0)  # std ≈ 0.37
        
        std = log_std.exp()
        
        # Sample using reparameterization trick
        try:
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Sample with reparameterization
        except Exception as e:
            print(f"⚠️ Sampling failed: {e}, using random fallback")
            x_t = torch.randn_like(mean) * std + mean
        
        # Tanh squashing to map to [-1, 1], then shift to [0, 1]
        y_t = torch.tanh(x_t)
        action = (y_t + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        
        # Compute log probability with change of variables formula
        # log π(a|s) = log μ(u|s) - log|da/du|
        # where a = (tanh(u) + 1) / 2
        log_prob = normal.log_prob(x_t)
        
        # Correction for tanh squashing: log(1 - tanh²(x))
        # and scaling factor 1/2 from the [0,1] transformation
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob -= torch.log(torch.tensor(2.0))  # Correction for scaling to [0,1]
        
        # Sum over action dimensions and clamp for stability
        log_prob = log_prob.sum(dim=1, keepdim=True)
        log_prob = torch.clamp(log_prob, -100, 100)
        
        # No hard clamp on action to preserve gradients
        # Action naturally stays in [0, 1] due to tanh transformation
        return action, log_prob

class Actor_broken(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor_broken, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.min_log_prob = -20
        self.max_log_prob = 2
        self.action_dim = action_dim

        
    def forward(self, state):
        if torch.isnan(state).any():
            print("Warning: Input state contains NaN!")
            state = torch.zeros_like(state)
            
        logits = self.net(state)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: Actor output is abnormal! Force recovery...")
            logits = torch.zeros_like(logits)
            
        return logits

    def sample(self, state, temperature=1.0):
        """Sample action using Gumbel-Softmax reparameterization"""
        logits = self.forward(state)

        # Gumbel-Softmax reparameterization sampling
        try:
            # temperature controls the exploration level (can be adjusted dynamically during training)
            action_probs = F.gumbel_softmax(
                logits, 
                tau=temperature, 
                hard=True  # generate one-hot vector
            )
            action = torch.argmax(action_probs, dim=-1).long() + 1  # convert to 1~10

        except Exception as e:
            print(f"Sampling failed: {str(e)}")
            action = torch.randint(1, self.action_dim+1, (logits.shape[0],)).to(logits.device)
            action_probs = F.one_hot(action-1, num_classes=self.action_dim).float()

        # Log probability calculation (with safety limits)
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = torch.clamp(log_prob, self.min_log_prob, self.max_log_prob)
        log_prob = (action_probs * log_prob).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Actor(nn.Module):
    """
    Improved Actor for discrete action space using Categorical distribution.
    
    Advantages over original Actor:
    1. Correct gradient flow - no gradient detachment issues
    2. Proper log probability calculation matching the actual sampling distribution
    3. True deterministic mode for inference
    4. Cleaner and more standard implementation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)  # Output logits for each action
        )
        
        self.action_dim = action_dim
        self.min_log_prob = -20  # Numerical stability
        self.max_log_prob = 2
        
    def forward(self, state):
        """
        Forward pass to get action logits
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            logits: Action logits [batch_size, action_dim]
        """
        # Input validation
        if torch.isnan(state).any():
            print("⚠️ Warning: Input state contains NaN! Using zeros.")
            state = torch.zeros_like(state)
        
        logits = self.net(state)
        
        # Output validation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠️ Warning: Actor output is abnormal! Using uniform distribution.")
            logits = torch.zeros_like(logits)
            
        return logits
    
    def sample(self, state, deterministic=False):
        """
        Sample action from the policy
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, select argmax action (for inference)
                          If False, sample from distribution (for training)
            
        Returns:
            action: Sampled action [batch_size] with values in [1, action_dim]
            log_prob: Log probability of the action [batch_size, 1]
        """
        logits = self.forward(state)
        
        try:
            # Create Categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                # Deterministic mode: select action with highest probability
                action_idx = torch.argmax(logits, dim=-1)
            else:
                # Stochastic mode: sample from distribution
                action_idx = dist.sample()
            
            # Calculate log probability
            log_prob = dist.log_prob(action_idx)
            
            # Clamp log_prob for numerical stability
            log_prob = torch.clamp(log_prob, self.min_log_prob, self.max_log_prob)
            log_prob = log_prob.unsqueeze(-1)  # [batch_size, 1]
            
            # Convert to action range [1, action_dim]
            action = action_idx + 1
            
            return action, log_prob
            
        except Exception as e:
            print(f"❌ Sampling failed: {str(e)}, using random action")
            # Fallback: random uniform action
            action = torch.randint(
                1, self.action_dim + 1, 
                (logits.shape[0],), 
                device=logits.device
            )
            # Uniform log probability
            log_prob = torch.full(
                (logits.shape[0], 1), 
                -np.log(self.action_dim), 
                device=logits.device
            )
            return action, log_prob
    
    def get_log_prob(self, state, action):
        """
        Get log probability of a specific action (useful for off-policy learning)
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size] with values in [1, action_dim]
            
        Returns:
            log_prob: Log probability [batch_size, 1]
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Convert action from [1, action_dim] to [0, action_dim-1]
        action_idx = action - 1
        
        log_prob = dist.log_prob(action_idx)
        log_prob = torch.clamp(log_prob, self.min_log_prob, self.max_log_prob)
        
        return log_prob.unsqueeze(-1)
    
    def get_action_probs(self, state):
        """
        Get probability distribution over actions
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            probs: Action probabilities [batch_size, action_dim]
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class Value(nn.Module):
    def __init__(self, state_dim, hidden_state=256, dff=256):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.hidden_state = hidden_state
        self.dff = dff

        self.fc1 = nn.Linear(self.state_dim, self.hidden_state)
        self.fc2 = nn.Linear(self.hidden_state, self.dff)
        self.v = nn.Linear(self.dff, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        
        return v

class SpeculativeInferenceEnv:
    def __init__(self, 
                 total_power=10,  # total power(W)
                 max_depth=10,
                 history_size=10,
                 alpha=0.03,  # ratio between depth and computation time
                 B=5e5,       # Channel bandwidth (Hz)
                 Omega=1,     # Scale Parameter of nakagami
                 m=0.5,       # Fading Parameter of nakagami
                 sigma2=1e-5): # noise power
        """
        Spculative decoding environment
        
        Parameters:
        total_power: total power of edge device (W)
        max_depth: maximum depth
        history_size: history length of accpet length
        """
        self.total_power = total_power
        self.max_depth = max_depth
        self.alpha = alpha
        self.B = B
        self.H = 1
        self.omega = Omega
        self.m = m
        self.sigma2 = sigma2
        self.comm_time = None
        self.comp_time = None

        self.state_dim = 6
        
        # action dimension: [depth_ratio, comm_power_ratio]
        self.action_dim = 2
        
        # history
        self.history_size = history_size
        self.accept_length_history = deque(maxlen=history_size)
        self.accept_ratio_history = deque(maxlen=history_size)
        self.iter_time_history = deque(maxlen=history_size)
        self.accept_length_history_gradient = deque(maxlen=history_size)
        self.accept_ratio_history_gradient = deque(maxlen=history_size)

        # current state
        self.current_depth = 7  # default depth
        self.comm_power = total_power * 0.3  # initial communication power allocation
        self.comp_power = total_power - self.comm_power
        self.topk = 10  # default topk

    def reset(self):
        self.accept_length_history.clear()
        self.accept_ratio_history.clear()
        self.iter_time_history.clear()
        self.accept_length_history_gradient.clear()
        self.accept_ratio_history_gradient.clear()
        self.current_depth = 7
        self.comm_power = self.total_power * 0.3
        self.comp_power = self.total_power - self.comm_power
        return self._get_state()

    def update_state(self, accept_length, accept_ratio, iter_time):
        self.accept_length_history.append(accept_length)
        self.accept_ratio_history.append(accept_ratio)
        self.iter_time_history.append(iter_time)
        self.iter_time = iter_time
        if len(self.accept_length_history) > 3:
            depth_slope = self.calculate_last_three_slope(self.accept_length_history)
            ratio_slope = self.calculate_last_three_slope(self.accept_ratio_history)
            self.accept_length_history_gradient.append(depth_slope)
            self.accept_ratio_history_gradient.append(ratio_slope)
        else:
            self.accept_length_history_gradient.append(0)
            self.accept_ratio_history_gradient.append(0)

        return self._get_state()

    def _get_state(self):
        avg_length = np.mean(self.accept_length_history) if self.accept_length_history else 0
        avg_ratio = np.mean(self.accept_ratio_history) if self.accept_ratio_history else 0
        
        return np.array([
            self.accept_length_history[-1] if self.accept_length_history else 0,  # current accept length
            self.accept_ratio_history[-1] if self.accept_ratio_history else 0,  # current accept ratio
            avg_length,  # average accept length
            avg_ratio,  # average accept ratio
        ], dtype=np.float32)
    
    def channel_gain_nakagami(self):
        self.H = nakagami.rvs(self.m, scale=np.sqrt(self.omega/self.m), size=1)[0]
        print("Channel Gain: ", self.H)
    
    def calculate_iteration_time(self):
        """
        Iteration time(model inference time + communication time)
        
        Parameters:
        depth: depth of speculative decoding (1 to max_depth)
        comm_power_ratio: ratio between total power and communication power (0-1)
        
        Return:
        iter_time: iteration time in second
        """
        print("Comp powerer:", self.comp_power, "Comm power:", self.comm_power)

        self.comp_time = 1 if self.comp_power < 0.1 else (self.alpha * (self.current_depth + 1)) / self.comp_power
        
        # data size to be transfered (bits)
        if self.current_depth * self.topk + 1 < 60:
            data_size = 128*(1 + (self.current_depth + 1)*self.topk) + ((self.current_depth + 1) * self.topk)**2 * 32 + 14000
        else:
            data_size = 128 * 60 + 60**2 * 32 + 100000
        
        snr = (self.H**2 * self.comm_power) / max(self.sigma2, 1e-12)
        communication_rate = self.B * np.log2(1 + snr) if snr > 0 else 0
        
        self.comm_time = 1 if communication_rate < 1e-9 else data_size / communication_rate
        
        return self.comp_time + self.comm_time + 0.1
    
    
    def calculate_reward(self):

        return 1


    def step(self, action):

        # reward calculation
        # avg_accept = np.mean(self.accept_length_history) if self.accept_length_history else 0
        # avg_time = np.mean([t for t in self.iter_time_history]) if hasattr(self, 'iter_time_history') else 1e-6
        # reward = 0.5*self.accept_length_history[-1] + self.current_depth*self.accept_ratio_history[-1] + 2000*self.iter_time  # tokens/s
        # average of recent 3 accept lengths / iteration time
        # if self.comm_power < 0.1 or self.comp_power < 0.1:
        #     reward = -150

        # reward = min(200, np.mean(np.array(self.accept_length_history)[-2:]) / self.iter_time + self.current_depth * 10 - 1000*abs(self.comp_time-self.comm_time)\
        #     if len(self.accept_length_history) > 3 else self.accept_length_history[-1] / self.iter_time + self.current_depth * 10 - 1000*abs(self.comp_time-self.comm_time))
        
        # if abs(self.comp_power-self.comm_power) > 9.9:
        #     reward = -40
        # else:
        #     reward = 20*self.accept_length_history[-1]/np.mean(np.array(self.accept_length_history)) - self.iter_time*50 - 50*abs(self.comm_time-self.comp_time) + 15*self.accept_ratio_history[-1] + 3*self.current_depth
        #     print("Reward before normalized: ", reward, "Depth: ", self.current_depth, "History: ", np.mean(np.array(self.accept_length_history)))
        #     reward = max(min(reward,40),-40)

        # reward = reward / 40
        reward = self.calculate_reward()
        reward = reward/10
        
        return self._get_state(), reward, False #, info
    
    def calculate_last_three_slope(self, data):
        if len(data) < 3:
            return 0.0
        
        last_three = [data[-3], data[-2], data[-1]]
        x = range(3)
        slope, _, _, _, _ = linregress(x, last_three)
        return slope


class SACAgent:
    def __init__(self, state_dim, action_dim, device, 
                 gamma=0.99, tau=0.005, alpha=0.7, 
                 lr=3e-5, hidden_dim=256, buffer_size=100000):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        self.alpha_loss = None
        
        # Networks - Use Actor_1d for continuous action space (threshold in [1.0, 1.5])
        self.actor = Actor_1d(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr/3)
        self.target_critic_optimizer = optim.Adam(self.target_critic.parameters(), lr=lr/3)
        
        # Temperature (alpha) optimization
        self.target_entropy = -0.3 # heuristic value
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.replay_buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_count = 0
        self.replay_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.replay_actions = np.zeros((buffer_size, action_dim), dtype=np.float32)  # Changed to float32 for continuous action
        self.replay_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.replay_next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.replay_dones = np.zeros(buffer_size, dtype=np.bool_)

    def save_experience(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        if isinstance(action, (list, np.ndarray)) and len(action) == 1:
            action = action[0]
        action = np.array([float(action)], dtype=np.float32)  # Changed to float for continuous action
        next_state = np.array(next_state, dtype=np.float32)
        
        idx = self.buffer_ptr % self.replay_buffer_size
        self.replay_states[idx] = state
        self.replay_actions[idx] = action
        self.replay_rewards[idx] = reward
        self.replay_next_states[idx] = next_state
        self.replay_dones[idx] = done
        
        self.buffer_ptr = (self.buffer_ptr + 1) % self.replay_buffer_size
        self.buffer_count = min(self.buffer_count + 1, self.replay_buffer_size)


    def update(self, batch_size):
        # check if buffer has enough samples
        if self.buffer_count < batch_size:
            print(f"Warning: Buffer has only {self.buffer_count} samples (needs {batch_size})")
            return (0.0, 0.0, 0.0), False
        
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)  # Changed to FloatTensor for continuous action
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        self.actor.train()
        self.critic.train()

        # 1. Critic update (Q1, Q2)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_log_probs = torch.clamp(next_log_probs, -100, 100)
            # No one-hot encoding for continuous action space
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            target_q = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = torch.clamp(target_q, -1e4, 1e4)
            q_target = rewards + (1 - dones) * self.gamma * target_q

        torch.set_grad_enabled(True)  
        # Continuous action - directly use action values
        q1, q2 = self.critic(states, actions)
        q1_loss = F.mse_loss(q1, q_target.detach()) + 1e-6
        q2_loss = F.mse_loss(q2, q_target.detach()) + 1e-6
        q_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # avoid exploding gradients
        self.critic_optimizer.step()

        # 2. Actor update (π)
        # Actor update（freeze Critic）

        new_actions, log_probs = self.actor.sample(states)
        log_probs = torch.clamp(log_probs, -100, 100)
        with torch.no_grad():
            q1_new, q2_new = self.critic(states, new_actions)  # No one-hot encoding for continuous action
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()


        # 3. update alpha
        alpha_loss = -(
            self.log_alpha.to(log_probs.device) * 
            (log_probs.detach() + 
            torch.tensor(self.target_entropy, device=log_probs.device))
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 4. soft update target critic network
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return (q_loss.item(), actor_loss.item(), alpha_loss.item()), True


    def select_action(self, state, deterministic=False):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            action, _ = self.actor.sample(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'target_critic_optimizer': self.target_critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            
            'replay_buffer': {
                'states': self.replay_states,
                'actions': self.replay_actions,
                'rewards': self.replay_rewards,
                'next_states': self.replay_next_states,
                'dones': self.replay_dones,
                'ptr': self.buffer_ptr,
                'count': self.buffer_count
            }
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        filename = os.path.abspath(filename)
        print(f"Loading model from: {filename}")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Path does not exist: {filename}")
        if os.path.isdir(filename):
            raise IsADirectoryError(f"Expected file path but got directory: {filename}")
        if not filename.endswith(('.pth', '.pt')):
            print(f"Warning: File extension suggests this may not be a PyTorch model: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                checkpoint = torch.load(f,weights_only=False)
            # Network parameters
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            # optimizer states
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.target_critic_optimizer.load_state_dict(checkpoint['target_critic_optimizer'])
            
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            
            # Buffer data
            if 'replay_buffer' in checkpoint:
                buffer_data = checkpoint['replay_buffer']
                self.replay_states = buffer_data['states']
                self.replay_actions = buffer_data['actions']
                self.replay_rewards = buffer_data['rewards']
                self.replay_next_states = buffer_data['next_states']
                self.replay_dones = buffer_data['dones']
                self.buffer_ptr = buffer_data['ptr']
                self.buffer_count = buffer_data['count']
                
            print(f"Successfully loaded model from {filename}")
            print("Size of replay buffer:", self.buffer_count)
            return True
            
        except PermissionError:
            print(f"[Error] Permission denied. Try:")
            print(f"1. Close the file if opened in other programs")
            print(f"2. Run as administrator")
            print(f"3. Check file permissions")
            raise
        except Exception as e:
            print(f"[Error] Load failed: {type(e).__name__}: {str(e)}")
            raise

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from the replay buffer"""
        # check if buffer has enough samples
        if self.buffer_count < batch_size:
            raise ValueError(f"Not enough samples ({self.buffer_count} < {batch_size})")
        
        # randomly select indices
        indices = np.random.choice(self.buffer_count, batch_size, replace=False)
        
        return (
            self.replay_states[indices],
            self.replay_actions[indices],
            self.replay_rewards[indices],
            self.replay_next_states[indices],
            self.replay_dones[indices]
        )
    
    
    def get_timestamped_path(self, base_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_path, f"sac_agent_{timestamp}.pth")





class SACAgent_fixTemperature: # No temperature learning
    def __init__(self, state_dim, action_dim, device, 
                 gamma=0.99, tau=0.005, alpha=0.5, 
                 lr=3e-5, hidden_dim=256, buffer_size=100000):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        self.alpha_loss = None
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.value = Value(state_dim, action_dim, hidden_dim).to(device)
        self.target_value = Value(state_dim, action_dim, hidden_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr/3)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.target_value_optimizer = optim.Adam(self.target_value.parameters(), lr=lr)
        
        # Temperature (alpha) optimization
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.replay_buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_count = 0
        self.replay_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.replay_actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.replay_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.replay_next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.replay_dones = np.zeros(buffer_size, dtype=np.bool_)

    def save_experience(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        idx = self.buffer_ptr % self.replay_buffer_size
        self.replay_states[idx] = state
        self.replay_actions[idx] = action
        self.replay_rewards[idx] = reward
        self.replay_next_states[idx] = next_state
        self.replay_dones[idx] = done
        
        self.buffer_ptr = (self.buffer_ptr + 1) % self.replay_buffer_size
        self.buffer_count = min(self.buffer_count + 1, self.replay_buffer_size)

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if self.buffer_count < batch_size:
            raise ValueError(f"Not enough samples ({self.buffer_count} < {batch_size})")
            
        indices = np.random.choice(self.buffer_count, batch_size, replace=False)
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices]
        )

    def update_target_network(self):
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def update(self, batch_size):
        # check if buffer has enough samples
        if self.buffer_count < batch_size:
            print(f"Warning: Buffer has only {self.buffer_count} samples (needs {batch_size})")
            return (0.0, 0.0, 0.0), False
        
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        self.actor.train()
        self.critic.train()
        self.value.train()

        # 1. Value update (V)
        torch.set_grad_enabled(True)
        value_actions, value_log_probs = self.actor.sample(states)
        q1_tensor, q2_tensor = self.critic(states, actions)
        q12 = torch.min(q1_tensor, q2_tensor)
        value_target = q12 - value_log_probs
        value = self.value(states)
        self.value_optimizer.zero_grad()
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())
        value_loss.backward()
        self.value_optimizer.step()

        # 2. Critic update (Q1, Q2)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_log_probs = torch.clamp(next_log_probs, -100, 100)
            q1_next = self.target_value(next_states)
            target_q = rewards + (1 - dones) * self.gamma * q1_next
            target_q = torch.clamp(target_q, -1e4, 1e4)

        torch.set_grad_enabled(True)  
        q1, q2 = self.critic(states, actions)
        q1_loss = F.mse_loss(q1, target_q.detach()) + 1e-6
        q2_loss = F.mse_loss(q2, target_q.detach()) + 1e-6
        q_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. Actor update (π)
        # Actor update（freeze Critic）
        for p in self.critic.parameters():
            p.requires_grad = False

        new_actions, log_probs = self.actor.sample(states)
        log_probs = torch.clamp(log_probs, -100, 100)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # unfreeze Critic
        for p in self.critic.parameters():
            p.requires_grad = True

        # 4. soft update target value network (V_target)
        with torch.no_grad():
            for param, target_param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return (q_loss.item(), actor_loss.item(), value_loss.item()), True

    def select_action(self, state, deterministic=False):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            action, _ = self.actor.sample(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'value': self.value.state_dict(),
            'target_value': self.target_value.state_dict(),
            
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            
            'replay_buffer': {
                'states': self.replay_states,
                'actions': self.replay_actions,
                'rewards': self.replay_rewards,
                'next_states': self.replay_next_states,
                'dones': self.replay_dones,
                'ptr': self.buffer_ptr,
                'count': self.buffer_count
            }
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        filename = os.path.abspath(filename)
        print(f"Loading model from: {filename}")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Path does not exist: {filename}")
        if os.path.isdir(filename):
            raise IsADirectoryError(f"Expected file path but got directory: {filename}")
        if not filename.endswith(('.pth', '.pt')):
            print(f"Warning: File extension suggests this may not be a PyTorch model: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                checkpoint = torch.load(f)
            # Network parameters
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.value.load_state_dict(checkpoint['value'])
            self.target_value.load_state_dict(checkpoint['target_value'])
            # optimizer states
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            
            # Buffer data
            if 'replay_buffer' in checkpoint:
                buffer_data = checkpoint['replay_buffer']
                self.replay_states = buffer_data['states']
                self.replay_actions = buffer_data['actions']
                self.replay_rewards = buffer_data['rewards']
                self.replay_next_states = buffer_data['next_states']
                self.replay_dones = buffer_data['dones']
                self.buffer_ptr = buffer_data['ptr']
                self.buffer_count = buffer_data['count']
                
            print(f"Successfully loaded model from {filename}")
            print("Size of replay buffer:", self.buffer_count)
            return True
            
        except PermissionError:
            print(f"[Error] Permission denied. Try:")
            print(f"1. Close the file if opened in other programs")
            print(f"2. Run as administrator")
            print(f"3. Check file permissions")
            raise
        except Exception as e:
            print(f"[Error] Load failed: {type(e).__name__}: {str(e)}")
            raise

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from the replay buffer"""
        # check if buffer has enough samples
        if self.buffer_count < batch_size:
            raise ValueError(f"Not enough samples ({self.buffer_count} < {batch_size})")
        
        # randomly select indices
        indices = np.random.choice(self.buffer_count, batch_size, replace=False)
        
        return (
            self.replay_states[indices],
            self.replay_actions[indices],
            self.replay_rewards[indices],
            self.replay_next_states[indices],
            self.replay_dones[indices]
        )
    
    
    def get_timestamped_path(self, base_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_path, f"sac_agent_{timestamp}.pth")
    
    def save_with_version(self, base_dir="", max_versions=5):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        existing = [f for f in os.listdir(base_dir) if f.startswith("sac_agent_")]
        existing.sort()
        
        while len(existing) >= max_versions:
            os.remove(os.path.join(base_dir, existing.pop(0)))
        
        save_path = self.get_timestamped_path(base_dir)
        self.save(save_path)
        return save_path
    

