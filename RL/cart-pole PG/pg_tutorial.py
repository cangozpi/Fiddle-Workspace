# ==============================================================
# Following the tutorial at https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn


# Deep Policy Network
class policy_estimator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        
        # Define network
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 16), 
            torch.nn.ReLU(), 
            torch.nn.Linear(16, self.n_outputs), # Note that there are possible actions many output dimensions
            torch.nn.Softmax(dim=-1)) # Softmax is used to return a Probability distribution over the possible action space
    
    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


# Function to return cumulative discounted rewards while calculating Return/Goal
def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean() # subtract mean for stability purposes this is not mandatory though


# REINFORCE Algorithm
def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = torch.optim.Adam(policy_estimator.parameters(), 
                           lr=0.01)
    
    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        env.render()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator(
                s_0).detach().numpy()
            action = np.random.choice(action_space, 
                p=action_probs)
            s_1, r, done, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor(
                       batch_actions)
                    
                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator(state_tensor)) # --> [B, #episodes, env.action_space.n]
                    selected_logprobs = reward_tensor * \
                        torch.gather(logprob, 1, 
                        action_tensor.unsqueeze(-1)).squeeze() # = pi(a|s) * R
                    loss = -selected_logprobs.mean() # = -E[ pi(a|s) * R ]
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print(f"\rEp: {ep + 1} Average reward of last 100 episodes = {avg_rewards:.2f}", end="")
                ep += 1
                
    return total_rewards 


# Function to plot Rewards
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.plot(np.array(rewards).mean())
    plt.title("REINFORCE Training Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.show()


# Run the code
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    policy_est = policy_estimator(env)
    env.close()
    rewards = reinforce(env, policy_est)
    plot_rewards(rewards)

# Note that CartPole is solved after an average score of 195 or more for 100 consecutive episodes)

