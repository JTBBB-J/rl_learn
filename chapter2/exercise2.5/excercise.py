import matplotlib
import matplotlib.pyplot as plt
import random
from tqdm import trange
import numpy as np

class bandit:
    # q_real 代表真实value 值
    # q_estimate 代表value 观测值
    def __init__(self,k_arm=10,step_size=0,varepsilon=0.1,times=1,station = True,alpha = 0):
        self.k = k_arm
        self.varepsilon = varepsilon        #非贪婪操作概率
        self.times = times
        self.station = station
        self.alpha = alpha
    def reset(self):
        self.times = 1
        #题中写道q*设置相同初始值
        self.q_real = np.zeros(self.k)+np.random.normal(1,10,self.k)
        self.q_estimate =np.zeros(self.k)
        #实际最佳行动
        self.best_action = np.argmax(self.q_real)
        self.action_count = np.zeros(self.k)

    ##确定哪个臂执行行动    
    def action(self):
        if np.random.rand() < self.varepsilon:
            return random.randint(1,10)-1

        greedy_action =  np.argmax(self.q_estimate)
        return greedy_action

    #更新估计值和真实值
    def step(self,action_arm):
        #获得reward
        self.action_count[action_arm] +=1
        reward = self.q_real[action_arm] + np.random.normal(loc=0,scale=2)
        if self.alpha:
           self.q_estimate[action_arm] += (reward - self.q_estimate[action_arm])*self.alpha
        else:
           self.q_estimate[action_arm] += (reward - self.q_estimate[action_arm])/self.action_count[action_arm] 
        self.times += 1
        if not self.station:
           self.q_real += np.random.normal(0,0.01,self.k)
           self.best_action = np.argmax(self.q_real)

        return reward


# runs 一共跑几轮 times一轮跑几次
def simulate(runs,times,bandits):
    rewards = np.zeros((len(bandits),runs,times))
    best_action_count = np.zeros(rewards.shape)
    for i,bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(times):
                action = bandit.action()
                reward = bandit.step(action)
                rewards[i,r,t] = reward
                if action == bandit.best_action:
                    best_action_count[i,r,t] = 1
    mean_best_action_counts = best_action_count.mean(axis=1)
    mean_reward = rewards.mean(axis =1)
    return mean_best_action_counts,mean_reward

def plot(reward,count,rew_name,cou_name):
    plt.figure(figsize=(10,20))
    plt.plot(reward[0])
    plt.xlabel('steps')
    plt.ylabel('average reward')
    #plt.legend()
    plt.savefig('images//'+rew_name+'.png')  #在plt.show()后调用会保存空白的
    plt.show()


    plt.figure(figsize=(10,20))
    plt.plot(count[0])
    plt.xlabel('steps')
    plt.ylabel('best action count')
    plt.savefig('images//'+cou_name+'.png')
    plt.show()



def fig1(runs=2000,times=1000):
    epsilon = 0.1
    bandits = [bandit(varepsilon=epsilon)]
    best_action_count,rewards=simulate(runs,times,bandits)
    plot(reward=rewards,count=best_action_count,rew_name='fig_excercise2.5_stationary_nonstepsizeconst_reward',cou_name='excercise2.5_stationary_nonstepsizeconst_bestaction')
    
def fig2(runs=2000,times=1000):
    epsilon = 0.1
    bandits = [bandit(varepsilon=epsilon,station=False)]
    best_action_count,rewards=simulate(runs,times,bandits)
    plot(reward=rewards,count=best_action_count,rew_name='fig_excercise2.5_nonestationary_nonstepsizeconst_reward',cou_name='excercise2.5_nonestationary_nonstepsizeconst_bestaction')
    

def fig3(runs=2000,times=1000):
    epsilon = 0.1
    bandits = [bandit(varepsilon=epsilon,station=False,alpha=0.1)]
    best_action_count,rewards=simulate(runs,times,bandits)
    plot(reward=rewards,count=best_action_count,rew_name='fig_excercise2.5_nonestationary_stepsizeconst_reward',cou_name='excercise2.5_nonestationary_stepsizeconst_bestaction')
  



if __name__ == "__main__" :
    print("1")
    fig3()