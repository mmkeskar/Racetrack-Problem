#!/usr/bin/env python
# coding: utf-8


#%%
import numpy as np
import random
import matplotlib.pyplot as plt




class RaceTrack1:
  
    terminal_states = [(16, 26), (16, 27), (16, 28), (16, 29), (16, 30), (16, 31)]
    starting_states = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
    
    boundary = [(3, 0), (3, 3), (2, 3), (2, 10), (1, 10), (1, 18), (0, 18), (0, 28), (1, 28), (1, 29), (2, 29), (2, 31), (3, 31), (3, 32), (17, 32), (17, 26), (10, 26), (10, 25), (9, 25), (9, 0)]
  
    @staticmethod
    def on_track(s):
        if s[0] < 0 or s[0] > 16 or s[1] < 0 or s[1] > 31:
            return False
        if s[0] == 0 and (s[1] < 18 or s[1] > 27):
            return False
        if s[0] == 1 and (s[1] < 10 or s[1] > 28):
            return False
        if s[0] == 2 and (s[1] < 3 or s[1] > 30):
            return False
        if s[0] == 9 and s[1] < 25:
            return False
        if 10 <= s[0] <= 16 and s[1] < 26:
            return False
        return True
  
class RaceTrack3:
  
    terminal_states = [(16, 26), (16, 27), (16, 28), (16, 29), (16, 30), (16, 31)]
    starting_states = [(8, 0)]
    
    boundary = [(3, 0), (3, 3), (2, 3), (2, 10), (1, 10), (1, 18), (0, 18), (0, 28), (1, 28), (1, 29), (2, 29), (2, 31), (3, 31), (3, 32), (17, 32), (17, 26), (10, 26), (10, 25), (9, 25), (9, 0)]
  
    @staticmethod
    def on_track(s):
        if s[0] < 0 or s[0] > 16 or s[1] < 0 or s[1] > 31:
            return False
        if s[0] == 0 and (s[1] < 18 or s[1] > 27):
            return False
        if s[0] == 1 and (s[1] < 10 or s[1] > 28):
            return False
        if s[0] == 2 and (s[1] < 3 or s[1] > 30):
            return False
        if s[0] == 9 and s[1] < 25:
            return False
        if 10 <= s[0] <= 16 and s[1] < 26:
            return False
        return True



class RaceTrack2:
  
    terminal_states = [(7, 4), (7, 5), (7, 6), (7, 7)]
    starting_states = [(0, 0)]
    
    boundary = [(0, 0), (0, 8), (8, 8), (4, 8), (4, 4), (4, 0)]
  
    @staticmethod
    def on_track(s):
        if s[0] < 0 or s[0] > 7 or s[1] < 0 or s[1] > 7:
            return False
        if 3 < s[0] <= 7 and s[1] < 4:
            return False
        return True




class RaceTrack:
  
    terminal_states = [(7, 4), (7, 5), (7, 6), (7, 7)]
    starting_states = [(0, 0), (1, 0), (2, 0), (3, 0)]
    
    boundary = [(0, 0), (0, 8), (8, 8), (4, 8), (4, 4), (4, 0)]
  
    @staticmethod
    def on_track(s):
        if s[0] < 0 or s[0] > 7 or s[1] < 0 or s[1] > 7:
            return False
        if 3 < s[0] <= 7 and s[1] < 4:
            return False
        return True
  


# RTDP algorithm:
# 
# Initialize S as one of the starting states, choose an action according to the greedy method. Observe the reward and the final state. Update state-value function as in chapter 4. Update the model with the deterministic next state and the reward. Then, for some number of steps, choose random states from the ones that have occurred and then, choose a greedy action and update the state value function again. Then, follow the on-policy method, where the greedy action and the next state become the current state and action in the next step of the episode. 
# 



class RTDP:

    epsilon = 0.1
    gamma = 0.9

    def __init__(self, track):
        self.track = track
        
        self.V = dict()
        self.model = dict()
        self.encountered_states = []
        self.action_limit = 1
        self.rewards = []
        self.actions = np.zeros((2*self.action_limit+1, 2*self.action_limit+1), dtype='object')
        for index, action in np.ndenumerate(self.actions):
            self.actions[index[0], index[1]] = (index[0]-self.action_limit, index[1]-self.action_limit)
        self.actions = self.actions.flatten().tolist()
        self.policy = dict()
  
    def get_next_state_reward(self, S, A):
        
        single_state = True
        S1 = (S[0]+S[2]+A[0], S[1]+S[3]+A[1], S[2]+A[0], S[3]+A[1])
        if not self.track.on_track(S1):
            state = random.choice(self.track.starting_states)
            S1 = (state[0], state[1], 0, 0)
            single_state = False
        R = -1
        if S1 in self.track.terminal_states:
            R = 0 
        return S1, R, single_state
            
    def epsilon_greedy(self, actions, S, epsilon=0):
        #print(actions)
        rand = np.random.choice([True, False], 1, p = [epsilon, 1-epsilon])
        A = random.choice(actions)
        if rand:
            return A
        else:
            same_return = []
            S1, r, _ = self.get_next_state_reward(S, A)

            if S1 not in self.V.keys():
                self.V[S1] = random.random()*2 - 1
            max_r = self.V[S1]
            for action in actions:
                # print(f"feeding this action into the function (from 80): {action}")
                S1, r, _ = self.get_next_state_reward(S=S, A=action)
                if S1 not in self.V.keys():
                    self.V[S1] = random.random()*2 - 1
                if self.V[S1] > max_r:
                    max_r = self.V[S1]
                    A = action
                    same_return = [action]
                elif self.V[S1] == max_r:
                    same_return.append(action)
            if len(same_return) == 0 or len(same_return) == 1:
                return A
            else:
                return random.choice(same_return)
            
    def get_total_increment(self, S, A):
        S1, r, single_state = self.get_next_state_reward(S, A)
        if single_state:
            if S1 not in self.V.keys():
                self.V[S1] = random.random()*2 - 1
            self.V[S] = (r + self.gamma*self.V[S1])
        else:
            increment = 0
            for s in self.track.starting_states:
                if s not in self.V.keys():
                    self.V[s] = random.random()*2 - 1
                increment += (1.0/len(self.track.starting_states))*(r + self.gamma*self.V[s])
            self.V[S] = increment
            
    def rtdp(self, num_episodes, n):
        time = 0
        
        while time < num_episodes:
            time += 1
            state = self.track.starting_states[int(np.random.choice(range(len(self.track.starting_states)), 1, p=[1.0/len(self.track.starting_states) for s in self.track.starting_states]))]
            S = (state[0], state[1], 0, 0)
            A = self.epsilon_greedy(actions=self.actions, S=S)
            reward = 0
        
            step = 0
            
            while (S[0], S[1]) not in self.track.terminal_states:
                step+=1
                
                self.get_total_increment(S, A)    
                
                S1, R, _ = self.get_next_state_reward(S, A)
                if S not in self.model.keys():
                    self.model[S] = dict()
                if A not in self.model[S].keys():
                    self.model[S][A] = dict()
                    self.model[S][A]['times'] = 1
                else:
                    self.model[S][A]['times'] += 1
                if (S1, R) in self.model[S][A].keys():
                    self.model[S][A][(S1, R)] += 1
                else:
                    self.model[S][A][(S1, R)] = 1
    
                reward += R
                if S not in self.encountered_states:
                    self.encountered_states.append(S)

                for t in range(n):
                    sample_s = random.choice(self.encountered_states)
                    sample_a = self.epsilon_greedy(actions=list(self.model[sample_s].keys()), S=sample_s)
                    
                    possible_states = list(self.model[sample_s][sample_a].keys())
                    possible_states.remove('times')
                    probs = [self.model[sample_s][sample_a][pair]/self.model[sample_s][sample_a]['times'] for pair in possible_states]
                    
                    increment = 0
                    for i in range(len(possible_states)):
                        if possible_states[i][0] not in self.V.keys():
                            self.V[possible_states[i][0]] = random.random()*2 - 1
                        increment += probs[i]*(possible_states[i][1] + self.gamma*self.V[possible_states[i][0]])
                    self.V[sample_s] = increment    
                
                print(f"End of step {step} in episode {time}")
               

                S = S1
                A = self.epsilon_greedy(actions=self.actions, S=S)
                
            curr_state = (state[0], state[1], 0, 0)
            self.rewards.append(reward)
            rounds = 0
            while rounds < 30:
                action = self.epsilon_greedy(actions=self.actions, S=curr_state)
                print(f"Current state: {curr_state}, policy: {action}")
                rounds += 1
                curr_state, r, _ = self.get_next_state_reward(curr_state, action)
                if (curr_state[0], curr_state[1]) in self.track.terminal_states:
                    break

      
        for key in list(self.model.keys()):
            self.policy[key] = self.epsilon_greedy(actions=list(self.model[key].keys()), S=key)
        return self.V, self.model, self.rewards, self.policy
    
    def plot_rewards(self, num_episodes):
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.xlabel("Episodes")
        plt.ylabel("Sum of rewards in every episode")
        plt.plot(range(1, num_episodes+1), self.rewards)
        plt.show()
        
    def plot_path(self, bounds_x=(0, 10), bounds_y=(0, 10)):
        state = self.track.starting_states[int(np.random.choice(range(len(self.track.starting_states)), 1, p=[1.0/len(self.track.starting_states) for s in self.track.starting_states]))]
        S = (state[0], state[1], 0, 0)
        path = [(S[0]+0.5, S[1]+0.5, S[2], S[3])]
        t = 0
        while t < 50:
            if (S[0], S[1]) in self.track.terminal_states:
                break
            t += 1
            A = self.policy[S]
            print(f"Action we took: {A}")
            S, r, _ = self.get_next_state_reward(S, A)
            print(f"landed in {S}")
            path.append((S[0]+0.5, S[1]+0.5, S[2], S[3]))

        print(path)
        
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(20)
        plt.xlim([bounds_x[0], bounds_x[1]])
        plt.ylim([bounds_y[0], bounds_y[1]])
        
        for i in range(1, len(self.track.boundary)):
            x = [self.track.boundary[i-1][0], self.track.boundary[i][0]]
            y = [self.track.boundary[i-1][1], self.track.boundary[i][1]]
            plt.plot(x, y, color = 'k', linewidth=3) 

        for i in range(1, len(path)):
            plt.annotate(f"Point {i}", (path[i][0], path[i][1]))
            x = [path[i-1][0], path[i][0]]
            y = [path[i-1][1], path[i][1]]
            plt.plot(x, y)
            
        plt.xticks(range(bounds_x[1]+1))
        plt.yticks(range(bounds_x[1]+1))
        plt.grid(linewidth=1)
    

#%%


track1 = RaceTrack()
track2 = RaceTrack2()




agent = RTDP(track2)
V, model, rewards, policy = agent.rtdp(10000, 30)
agent.plot_rewards(10000)
agent.plot_path()




agent2 = RTDP(track2) 
V2, model2, rewards2, policy2 = agent2.rtdp(5, 30)
agent2.plot_rewards(5)
agent2.plot_path()




agent3 = RTDP(track1)
V3, model3, rewards3, policy3 = agent3.rtdp(8000, 150)
agent3.plot_rewards(8000)
agent3.plot_path()




moving_avg_1 = []
for i in range(100, len(rewards)):
    moving_avg_1.append(np.array(rewards[i-100:i]).mean())
    
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(10)
plt.plot(moving_avg_1)




moving_avg_2 = []
for i in range(20, len(rewards2)):
    moving_avg_2.append(np.array(rewards2[i-20:i]).mean())
    
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(10)
plt.plot(moving_avg_2)




moving_avg_3 = []
for i in range(100, len(rewards3)):
    moving_avg_3.append(np.array(rewards3[i-100:i]).mean())
    
f = plt.figure()
f.set_figwidth(20)
f.set_figheight(10)
plt.plot(moving_avg_3)




track4 = RaceTrack1()

agent4 = RTDP(track4)
V4, model4, rewards4, policy4 = agent4.rtdp(4000, 100)
agent4.plot_rewards(4000)
agent4.plot_path((0,35), (0, 35))




agent4.plot_rewards(4000)
agent4.plot_path((0,35), (0, 35))




agent5 = RTDP(track2)
V5, model5, rewards5, policy5 = agent5.rtdp(5000, 60)
agent5.plot_rewards(5000)
agent5.plot_path()




track6 = RaceTrack1()

agent6 = RTDP(track4)
V6, model6, rewards6, policy6 = agent6.rtdp(12000, 1800)
agent6.plot_rewards(12000)
agent6.plot_path((0,35), (0, 35))


agent7 = RTDP(track2)
V7, model7, rewards7, policy7 = agent7.rtdp(5000, 100)
agent7.plot_rewards(5000)
agent7.plot_path()



track8 = RaceTrack1()

agent8 = RTDP(track4)
V8, model8, rewards8, policy8 = agent8.rtdp(8000, 1500)
agent8.plot_rewards(8000)
agent8.plot_path((0,35), (0, 35))

agent9 = RTDP(track4)
V9, model9, rewards9, policy9 = agent9.rtdp(12000, 1800)
agent9.plot_rewards(12000)
agent9.plot_path((0,35), (0, 35))


track10 = RaceTrack3()
agent10 = RTDP(track10)
V10, model10, rewards10, policy10 = agent10.rtdp(10000, 300)
agent10.plot_rewards(10000)
agent10.plot_path((0, 35), (0, 35))


