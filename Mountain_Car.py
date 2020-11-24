import gym
slow = True

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
env.reset()

#Observe state and action space
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")

print(f"State space range: Low={env.observation_space.low}")
print(f"State space range: High={env.observation_space.high}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#define Q-learning function

def Qlearn(env, learn_rate, discount, epsilon, episodes):
    #discretize the state action space and determine its size
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    #for position we'll consider 19 states between 0.6 and -1.2
    #for velocity we'll consider 15 states between -0.07 and 0.07
    #total 19*15*3 = 855 state action values
    
    #iniitialze Q table
    Q = np.random.uniform(low =-1, high =1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    
    #initialize rewards
    reward_list = []
    avg_rewards = []
        
    #decaying epsilon each episode for better convergence
    #reduction is the decay after each episode
    reduction = epsilon/episodes
    
    #Q-learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize current state based on position and velocity
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        
        while done != True:
            #show results for last 10 episodes
            #if i >= (episodes - 10):
                #env.render()
            
            #determine next action with epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
            
            #next state, reward, action
            state2, reward, done, info = env.step(action)
    
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Updating Q table after next state determined
            #If next state is terminal state or 200 steps done then Q value = final reward
            #If not terminal state then update Q with 1-step return
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
            else:
                delta = learn_rate*(reward + discount*np.max(Q[state2_adj[0],state2_adj[1]])
                                    - Q[state_adj[0],state_adj[1],action])
                Q[state_adj[0],state_adj[1],action] += delta
                
            #update total reward and state
            tot_reward += reward
            state_adj = state2_adj
        
        #turn of epsilon greedy if total reward of an episode is <160
        #decay epsilon after end of episode
        if tot_reward < 160:
            epsilon = 0
        if epsilon > 0:
            epsilon -= reduction
        
        #Track reward for each episode
        reward_list.append(tot_reward)
        
        #Calculate average reward for 100 episodes
        if (i+1) % 100 == 0:
            avg_rewards.append(np.mean(reward_list))
            print(f"Episode: {i+1}. Average Reward: {np.mean(reward_list)}")
            reward_list = []
                    
    env.close()
    return avg_rewards, Q

#Run Q-learning algorithm
#run 1 no epsilon decay
rewards, Q_table = Qlearn(env, learn_rate=0.2, discount=0.9, epsilon=0.2, episodes=5000)
#run 2 discount = 1
rewards, Q_table = Qlearn(env, learn_rate=0.2, discount=1, epsilon=0.1, episodes=5000)
#run 3 with learning rate = 0.01
rewards, Q_table = Qlearn(env, learn_rate=0.01, discount=0.9, epsilon=0.1, episodes=8000)
#run 4 with epsilon = 0.8
rewards, Q_table = Qlearn(env, learn_rate=0.2, discount=0.9, epsilon=0.8, episodes=8000)
#run 5 with epsilon = 0 if total reward <160
rewards, Q_table = Qlearn(env, learn_rate=0.2, discount=0.9, epsilon=0.1, episodes=5000)

# Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1),rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Average Reward per 100 Episodes')
plt.show()

#Plot value function
value_function = np.max(Q_table, axis=2)
sns.heatmap(value_function,cmap = "crest", 
            yticklabels = np.round(np.arange(-1.2,0.7,0.1),1), 
            xticklabels = np.round(np.arange(-0.07,0.07,0.01),2))
