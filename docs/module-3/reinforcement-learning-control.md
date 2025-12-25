# Save this file as specify/implement/module-3/reinforcement-learning-control.md

# Reinforcement Learning for Robot Control

## Overview

Reinforcement learning (RL) enables robots to learn complex behaviors through interaction with the environment. NVIDIA Isaac provides tools for implementing RL algorithms with hardware acceleration, making it ideal for training humanoid robots with complex dynamics.

## Isaac Gym for RL Training

Isaac Gym provides a GPU-accelerated physics simulation environment for training RL agents:

```python
# Example RL environment using Isaac Gym
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np

class HumanoidRLEnv:
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        
        # Configure environment
        self.num_envs = cfg["env"]["numEnvs"]
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        
        # Initialize tensors
        self.obs_buf = torch.zeros((self.num_envs, cfg["env"]["numObservations"]), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        
        # Initialize sim
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        
        # Initialize tensors for GPU simulation
        self.acquire_tensors()
        
    def create_sim(self):
        # Create simulation
        self.sim = self.gym.create_sim(
            self.device_id, self.physics_engine, self.sim_params)
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        self.create_envs()
        
    def create_envs(self):
        # Load humanoid asset
        asset_root = "path/to/humanoid/asset"
        asset_file = "humanoid.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # Create environments
        spacing = self.cfg["env"]["envSpacing"]
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            
            # Add humanoid to environment
            humanoid_handle = self.gym.create_actor(
                env_ptr, humanoid_asset, self.start_positions[i], "humanoid", i, 1, 0)
                
            # Configure DOF properties
            dof_props = self.gym.get_actor_dof_properties(env_ptr, humanoid_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:] = 200.0
            dof_props["damping"][:] = 10.0
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_props)
            
            # Store handles
            self.humanoid_handles.append(humanoid_handle)
            self.envs.append(env_ptr)
            
    def acquire_tensors(self):
        # Acquire tensors for GPU simulation
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # Wrap tensors in PyTorch tensors
        self.root_states = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, -1, 13)
        self.dof_states = gymtorch.wrap_tensor(self.dof_tensor).view(self.num_envs, -1, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(self.rigid_body_tensor).view(self.num_envs, -1, 13)
        
    def compute_observations(self):
        # Compute observations for the agent
        # This is a simplified example - real implementation would include more complex features
        self.obs_buf[:, 0] = self.root_states[:, 0, 0]  # Position X
        self.obs_buf[:, 1] = self.root_states[:, 0, 1]  # Position Y
        self.obs_buf[:, 2] = self.root_states[:, 0, 2]  # Position Z
        self.obs_buf[:, 3] = self.root_states[:, 0, 7]  # Orientation X
        self.obs_buf[:, 4] = self.root_states[:, 0, 8]  # Orientation Y
        self.obs_buf[:, 5] = self.root_states[:, 0, 9]  # Orientation Z
        self.obs_buf[:, 6] = self.root_states[:, 0, 10] # Orientation W
        self.obs_buf[:, 7] = self.dof_states[:, 0, 0]   # Joint position
        self.obs_buf[:, 8] = self.dof_states[:, 0, 1]   # Joint velocity
        
    def compute_rewards(self):
        # Compute rewards for the agent
        # This is a simplified example - real implementation would have more complex reward shaping
        self.rew_buf[:] = 1.0  # Basic reward for staying alive
        self.rew_buf[:] += self.root_states[:, 0, 0] * 0.1  # Reward for forward progress
        
    def reset_idx(self, env_ids):
        # Reset environments
        positions = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device)
        
        self.dof_states[env_ids, :, 0] = positions
        self.dof_states[env_ids, :, 1] = velocities
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

def torch_rand_float(lower, upper, shape, device):
    # Generate random floats in PyTorch
    return (upper - lower) * torch.rand(shape, device=device) + lower
```

## PPO Implementation for Humanoid Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        shared_features = self.shared_layers(obs)
        
        # Actor
        action_mean = self.actor_mean(shared_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Critic
        value = self.critic(shared_features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, obs):
        action_mean, action_logstd, _ = self.forward(obs)
        action_std = torch.exp(action_logstd)
        
        # Sample action from normal distribution
        action = torch.normal(action_mean, action_std)
        
        # Compute log probability
        log_prob = -0.5 * (((action - action_mean) / action_std) ** 2 + 2 * action_logstd + np.log(2 * np.pi)).sum(1)
        
        return action, log_prob

class PPO:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 epochs=10, batch_size=64):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.actor_critic = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def update(self, obs, actions, rewards, log_probs, values, dones):
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(obs.device)
        advantages = returns - values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensor
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        
        # PPO update
        for _ in range(self.epochs):
            action_means, action_logstds, new_values = self.actor_critic(obs)
            
            # Compute new log probabilities
            new_log_probs = -0.5 * (((actions - action_means) / torch.exp(action_logstds)) ** 2 + 
                                    2 * action_logstds + np.log(2 * np.pi)).sum(1)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
```

## Training Loop

```python
def train_humanoid(env, ppo_agent, num_iterations=1000):
    """Training loop for humanoid robot control"""
    
    obs = env.reset()
    
    for iteration in range(num_iterations):
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        log_probs_batch = []
        values_batch = []
        dones_batch = []
        
        # Collect trajectories
        for step in range(env.cfg["env"]["max_episode_length"]):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy
            action, log_prob = ppo_agent.actor_critic.get_action(obs_tensor)
            value = ppo_agent.actor_critic(obs_tensor)[2]
            
            # Store in batch
            obs_batch.append(obs)
            actions_batch.append(action.squeeze().detach().numpy())
            log_probs_batch.append(log_prob.item())
            values_batch.append(value.item())
            
            # Take action in environment
            obs, reward, done, info = env.step(action.detach().numpy())
            
            rewards_batch.append(reward)
            dones_batch.append(done)
            
            if done:
                obs = env.reset()
                break
        
        # Update PPO agent
        ppo_agent.update(
            np.array(obs_batch),
            np.array(actions_batch),
            np.array(rewards_batch),
            np.array(log_probs_batch),
            np.array(values_batch),
            np.array(dones_batch)
        )
        
        # Log progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Average Reward: {np.mean(rewards_batch)}")
```

## NVIDIA TensorRT Integration

For deployment, integrate with TensorRT for optimized inference:

```python
import torch
import tensorrt as trt
import numpy as np

def optimize_with_tensorrt(model, input_shape):
    """Optimize RL model with TensorRT for deployment"""
    
    # Convert PyTorch model to ONNX
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, "model.onnx", 
                      input_names=["input"], output_names=["output"])
    
    # Create TensorRT builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open("model.onnx", 'rb') as model:
        parser.parse(model.read())
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20  # 1MB
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision
    
    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Save optimized engine
    with open("optimized_model.engine", "wb") as f:
        f.write(serialized_engine)
```

## Practical Exercise 6.1: Humanoid Locomotion Training

Implement a reinforcement learning algorithm to train a humanoid robot for locomotion:

1. Set up Isaac Gym environment for humanoid robot
2. Design reward function for locomotion tasks (walking forward, maintaining balance)
3. Implement PPO algorithm for training
4. Train the policy in simulation
5. Evaluate the learned policy in various scenarios
6. Optimize the model using TensorRT for deployment
7. Document the training process and results

Focus on:
- Balance maintenance during locomotion
- Energy efficiency
- Robustness to disturbances
- Generalization to different terrains

Create a training report with learning curves and performance metrics.

## References

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 23-30.