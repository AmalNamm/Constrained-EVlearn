import numpy as np
import random
from collections import namedtuple, deque
import copy
from torch.distributions.normal import Normal 
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

class TGeLU(nn.Module):
    def __init__(self, tl, tr, device, inplace:bool = False):
        super(TGeLU, self).__init__()
        self.inplace = inplace
        self.device = device
        self.tr = torch.tensor(tr).to(self.device)
        self.tl = torch.tensor(tl).to(self.device)
         
        
    def forward(self, input):
        dist = Normal(torch.zeros(input.shape).to(self.device),torch.ones(input.shape).to(self.device))
        
        cond1 = (input>=self.tr)
        cond2 = (0<=input)*(input<self.tr)
        cond3 = (self.tl<=input)*(input<0)
        cond4 = (input<self.tl)
        
        term1 = self.tr*dist.cdf(self.tr) + (input-self.tr)*(1-dist.cdf(input-self.tr))
        term2 = input*dist.cdf(input)
        term3 = input*(1-dist.cdf(input))
        term4 = self.tl*(1-dist.cdf(self.tl)) + (input-self.tl)*dist.cdf(input-self.tl)
        
        return cond1*term1 + cond2*term2 + cond3*term3 + cond4*term4
                
        
    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else " "
        return inplace_str

class PolicyNetwork(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_actions, 
                 action_space, 
                 action_scaling_coef, 
                 hidden_dim = [400,300],
                 init_w = 3e-3, 
                 log_std_min = -20, 
                 log_std_max = 2, 
                 epsilon = 1e-6):
        
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

        
class RegressionBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x = []
        self.y = []
        self.position = 0
    
    def push(self, variables, targets):
        if len(self.x) < self.capacity and len(self.x)==len(self.y):
            self.x.append(None)
            self.y.append(None)
        
        self.x[self.position] = variables
        self.y[self.position] = targets
        self.position = (self.position + 1) % self.capacity
    
    def __len__(self):
        return len(self.x)
    
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400,300], init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)
        self.ln1 = nn.LayerNorm(hidden_size[0])
        self.ln2 = nn.LayerNorm(hidden_size[1])
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256, 128], tgelu_range=None, gamma=0.99): #Updated to include gamma as a parameter
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
            tgelu_range (list): Range for TGeLU activation [min, max]. If None, calculated dynamically.
            gamma (float): Discount factor for calculating TGeLU range if not provided.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # If tgelu_range is not provided, calculate a reasonable default based on theoretical bounds
        if tgelu_range is None: #updated to include a tgelu range = max_reward/(1-discount_factor)
            # Estimate max value using max_reward/(1-gamma) with a safety factor
            # Assume max_reward ~= 10.0 (this is a hyperparameter to be tuned)
            max_reward = 10.0
            safety_factor = 2.0
            value_bound = safety_factor * max_reward / (1 - gamma)
            tgelu_range = [-value_bound, value_bound]
            print(f"Using calculated TGeLU range: {tgelu_range}")
            
        self.tgelu = TGeLU(tgelu_range[0], tgelu_range[1], device)
        # Input layer
        self.fc_layers = [nn.Linear(state_size, fc_units[0])]

        # Intermediate layers
        for i in range(1, len(fc_units)):
            self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

        # Output layer
        self.fc_layers.append(nn.Linear(fc_units[-1], action_size))

        # ModuleList to register the layers with PyTorch
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for fc in self.fc_layers[:-1]:
            x = self.tgelu(fc(x))
        return torch.tanh(self.fc_layers[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256, 128],tgelu_range=[-1,1]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Initial layer
        self.fc1 = nn.Linear(state_size, fc_units[0])

        # Concatenation layer (adding action_size to the width)
        self.fc2 = nn.Linear(fc_units[0] + action_size, fc_units[1] if len(fc_units) > 1 else 1)


        self.tgelu = TGeLU(tgelu_range[0], tgelu_range[1],device)
        # Additional layers if any
        self.fc_layers = []
        for i in range(1, len(fc_units) - 1):
            self.fc_layers.append(nn.Linear(fc_units[i], fc_units[i + 1]))

        # If there are more than 2 fc_units, the last fc_layer will output the Q-value.
        # Otherwise, fc2 is responsible for that.
        #if len(fc_units) > 2: #COMMENTED 13.05.2025
         #   self.fc_layers.append(nn.Linear(fc_units[-1], 1))

        # === Constraint Critic Final Layer Initialization ===
        # In your Critic class __init__ method, after defining the last layer:
        if len(fc_units) > 2: #ADDED 13.05.2025
            nn.init.constant_(self.fc_layers[-1].weight, 0.01)
            nn.init.constant_(self.fc_layers[-1].bias, 0.0)

        # ModuleList to register the layers with PyTorch
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.tgelu(self.fc1(state))

        # Concatenate the action values with the output from the previous layer
        x = torch.cat((x, action), dim=1)
        x = self.tgelu(self.fc2(x))

        for fc in self.fc_layers:
            x = self.tgelu(fc(x))

        return x


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.25, decay_factor=0.005):
        ...
        self.decay_factor = decay_factor
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.internal_state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.internal_state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.internal_state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for _ in range(len(x))])
        self.internal_state = x + dx
        #self.sigma *= self.decay_factor
        return self.internal_state

class ReplayBuffer1:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = [deque(maxlen=capacity) for _ in range(num_agents)]

    def push(self, state, action, reward, next_state, done):
        for i in range(self.num_agents):
            self.buffer[i].append((state[i], action[i], reward[i], next_state[i], done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = [], [], [], [], []
        for i in range(self.num_agents):
            # For each agent, get a batch of experiences
            batch = random.sample(self.buffer[i], batch_size)

            # For each agent's batch, separate the experiences into state, action, reward, next_state, done
            state_i, action_i, reward_i, next_state_i, done_i = zip(*batch)
            state.append(np.stack(state_i))
            action.append(np.stack(action_i))
            reward.append(np.stack(reward_i))
            next_state.append(np.stack(next_state_i))
            done.append(np.stack(done_i))

        print("In: sample ReplayBuffer1")
        return state, action, reward, next_state, done

    def __len__(self):
        return min(len(self.buffer[i]) for i in range(self.num_agents))


class ReplayBuffer2: #new replay buffer that accounts for constraints values
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = [deque(maxlen=capacity) for _ in range(num_agents)]

    def push(self, state, action, reward, next_state,constraint_value, done):
        for i in range(self.num_agents):
            self.buffer[i].append((state[i], action[i], reward[i], next_state[i],constraint_value[i], done))

    def sample(self, batch_size):
        state, action, reward, next_state,constraint_value, done = [], [], [], [], [], []
        for i in range(self.num_agents):
            # For each agent, get a batch of experiences
            batch = random.sample(self.buffer[i], batch_size)

            # For each agent's batch, separate the experiences into state, action, reward, next_state, done
            state_i, action_i, reward_i, next_state_i, cons_i, done_i = zip(*batch)

            # Convert tensors to CPU and then to numpy if they are tensors
            state_i = [s.cpu().numpy() if torch.is_tensor(s) else s for s in state_i]
            action_i = [a.cpu().numpy() if torch.is_tensor(a) else a for a in action_i]
            reward_i = [r.cpu().numpy() if torch.is_tensor(r) else r for r in reward_i]
            next_state_i = [ns.cpu().numpy() if torch.is_tensor(ns) else ns for ns in next_state_i]
            cons_i = [c.cpu().numpy() if torch.is_tensor(c) else c for c in cons_i]
            done_i = [d.cpu().numpy() if torch.is_tensor(d) else d for d in done_i]

            ###TO CHECK ABOVE Added it because of an error
            
            state.append(np.stack(state_i))
            action.append(np.stack(action_i))
            reward.append(np.stack(reward_i))
            
            constraint_value.append(np.stack(cons_i))
            next_state.append(np.stack(next_state_i)) 
            done.append(np.stack(done_i))

        #print("In sample ReplayBuffer2!")
        return state, action, reward, next_state, constraint_value, done

    def __len__(self):
        return min(len(self.buffer[i]) for i in range(self.num_agents))

