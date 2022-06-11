# ********************************
# Modified Tutorial from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Start tensorboard using: $tensorboard --logdir logs
# ********************************
from modulefinder import Module
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
from torch import nn # for nn.Module
import torchvision.transforms as T
import tensorflow as tf


# Possible actions: 'right', 'left'
# model input: difference of two states(image frames)
# CNNs are chosen

# Tensorboard
log_dir, run_name = "logs/", "cartpole_training"
tb_file_writer = tf.summary.create_file_writer(log_dir+run_name)

# create gym env
env = gym.make('CartPole-v0').unwrapped

# ================ forget about this part
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================


# ============ Replay Buffer Definition:

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward')) # (s_old, a, s_new, r)


# ======== Model Definition:
class PolicyNetwork(nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return ((size - kernel_size) // stride)  + 1 # ((N-F) / S) + 1 = output_dim
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w))) # width after passing through Conv2d for 3 times
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h))) # height after passing through Conv2d for 3 times
        linear_input_size = convw * convh * 32 # Flattened dimension for the Dense layer = C*W*H
        self.head = nn.Linear(linear_input_size, outputs) # Note that outputs would be the possible actions for Q values (i.e. 'right', 'left')
        self.head_activation = nn.Softmax(dim=1) # to convert probability distribution over the action space

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device) # utilize gpu
        x = torch.nn.functional.relu(self.bn1(self.conv1(x))) # (conv + batch_norm + relu)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x))) # (conv + batch_norm + relu)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x))) # (conv + batch_norm + relu)
        x = x.view(x.size(0), -1) # Flatten input for Dense layer --> [B, C*H*W]
        x = self.head(x) #--> [B, 2] (Note that 2 stands for the Q values for 'right' and 'left' actions)
        return self.head_activation(x) # --> [B, 2] (probabilities)

# ======================= 

# ================ Input Extraction 
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()]) # resizes the image and turns into a torch.Tensor


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order [C, H, W].
    screen = env.render(mode='rgb_array').transpose((2, 0, 1)) # --> [C, H, W]
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)] # Note that we do not clip width since it matters 
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width) # x location of the middle of the cart in the world
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range] # We always take a screenshot of the world where the cart is always in the middle of the screen
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

# uncomment below for visualization of the above functions
# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()
# ================

#  =============== TRAINING
BATCH_SIZE = 1 # update after every episode terminates!
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
lr = 1e-4

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape # --> [B, C, H, W]

# Get number of actions from gym action space
n_actions = env.action_space.n

# pi() network
policy_net = PolicyNetwork(screen_height, screen_width, n_actions).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
rewards_history = []

steps_done = 0


def select_action(state):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# =================
cur_ep_memory = []
# ================ Optimization Function Definition:
def optimize_model(episode_num):
    global times_updated
    if (episode_num %BATCH_SIZE) == 0: # optimize if certain number of batches are available per update 
        transitions = cur_ep_memory
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute log(pi(a|s)) 
        log_action_probs = torch.log(policy_net(state_batch).gather(1, action_batch.unsqueeze(-1)))
        
        # Compute R
        def calc_discounted_cumulative_rewards(rewards_batch, gamma=0.99):
            batched_cum_rewards = []
            for batch in rewards_batch:
                R = [gamma ** i * batch[i] for i in range(batch.shape[0])]
                R = np.array(R)
                R = R[::-1].cumsum()[::-1]
                R -= R.mean() # For stability purposes this is not mandatory for REINFORCE
                batched_cum_rewards.append(R)
            return torch.tensor(batched_cum_rewards)
        
        batched_cum_rewards = calc_discounted_cumulative_rewards(reward_batch.unsqueeze(0), GAMMA)

        # Compute E[ grad(log(pi(s | a))) * R ]
        loss = log_action_probs * batched_cum_rewards
        loss = - loss.mean()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # clip gradients for stability (vanishing/exploding gradients problem)
        optimizer.step()

        # log to Tensorboard
        mean_loss = float(loss.detach().cpu().numpy())
        with tb_file_writer.as_default():
            tf.summary.scalar('Training Loss per optimization', mean_loss, episode_num)

# ===============

# ============ Training Loop
num_episodes = 1_000
times_updated = 0
for i_episode in range(num_episodes):
    episode_rewards = []
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    cur_ep_memory = []
    for t in count():
        # Select and perform an action
        action = select_action(state).squeeze(1) # --> [1]
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
            # Store the transition in memory
            cur_ep_memory.append((state.detach().cpu(), action.detach().cpu(), reward.detach().cpu()))
            episode_rewards.append(reward.item())
        else:
            rewards_history.append(np.array(episode_rewards).mean())
            next_state = None

        # Move to the next state
        state = next_state

        if done:
            # Perform one step of the optimization (on the policy network)
            optimize_model(i_episode)
            # log training informations  at the end of each epoch.
            with tb_file_writer.as_default():
                tf.summary.scalar('episode reward', t+1, step=i_episode)
                tf.summary.scalar('Decayed EPSILON', eps_threshold, step=i_episode)
            episode_durations.append(t + 1)
            # plot_durations()
            break
    # Print running average
    avg_rewards = np.mean(rewards_history[-100:])
    print(f"\rEp: {i_episode + 1} Average reward of last 100 episodes = {avg_rewards:.2f}", end="")
    
print('Complete')
env.render()
env.close()
# plt.ioff()
# plt.show()
# ================