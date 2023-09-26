"""
You need to write code to plot the graphs as required in task2 of the problem statement:
    - You can edit any code in this file but be careful when modifying the simulation specific code. 
    - The simulation framework as well as the BernoulliBandit implementation for this task have been separated from the rest of the assignment code and is contained solely in this file. This will be useful in case you would like to collect more information from runs rather than just regret.
"""

import numpy as np
from multiprocessing import Pool
from task1 import Eps_Greedy, UCB, KL_UCB
import matplotlib.pyplot as plt

class BernoulliArmTask2:
  def __init__(self, p):
    self.p = p

  def pull(self, num_pulls=None):
    return np.random.binomial(1, self.p, num_pulls)

class BernoulliBanditTask2:
  def __init__(self, probs=[0.3, 0.5, 0.7],):
    self.__arms = [BernoulliArmTask2(p) for p in probs]
    self.__max_p = max(probs)
    self.__regret = 0

  def pull(self, index):
    reward = self.__arms[index].pull()
    self.__regret += self.__max_p - reward
    return reward

  def regret(self):
    return self.__regret
  
  def num_arms(self):
    return len(self.__arms)


def single_sim_task2(seed=0, ALGO=Eps_Greedy, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = BernoulliBanditTask2(probs=PROBS)
  algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def simulate_task2(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_task2,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets

def task2(algorithm, horizon, p1s, p2s, num_sims=50):
    """generates the data for task2
    """
    probs = [[p1s[i], p2s[i]] for i in range(len(p1s))]

    regrets = []
    for prob in probs:
        regrets.append(simulate_task2(algorithm, prob, horizon, num_sims))

    return regrets

if __name__ == '__main__':
  # Part A
  # p1s = np.ones(19) * 0.9 
  # p2s = np.linspace(0, 0.9, num = 19, endpoint = True)

  # regrets = task2(UCB, 30000, p1s, p2s) # 50 sims
  # print("Part A regret calculated")

  # fig1, ax = plt.subplots()

  # ax.xaxis.set_ticks_position('bottom')
  # ax.yaxis.set_ticks_position('left')

  # ax.set_xlim(0, 0.9)
  # ax.set_ylim(-50, 200)  
  # ax.grid(True, which='both', axis='both', linestyle='--')

  # ax.plot(p2s, regrets, 'blue')

  # ax.xaxis.set_label_coords(0.5, -0.075)
  # ax.yaxis.set_label_coords(-0.1, 0.5)
  # ax.set_xlabel('p2', fontsize = 12, loc = 'center')
  # ax.set_ylabel('Regret', fontsize = 12, loc = 'center')
  # ax.legend()

  # plt.title('Variation of Regret with p2 for UCB algorithm')
  # plt.savefig('regret_vs_p2_A.png',dpi=300)
  # plt.show()

  # Part B
  Delta = 0.1
  p2s = np.linspace(0, 0.9, num = 19, endpoint = True)
  p1s = p2s + Delta 

  regrets_UCB = task2(UCB, 30000, p1s, p2s) # 50 sim
  print("Part B UCB Regret calculated")
  regrets_KL_UCB = task2(KL_UCB, 30000, p1s, p2s) # 50 sims
  print("Part B KL_UCB Regret calculated")

  fig, (ax1, ax2) = plt.subplots(2)

  ax1.xaxis.set_ticks_position('bottom')
  ax1.yaxis.set_ticks_position('left')

  ax1.set_xlim(0, 0.9)
  ax1.set_ylim(-10, 200)  
  ax1.grid(True, which='both', axis='both', linestyle='--')

  ax1.plot(p2s, regrets_UCB, 'blue', label = 'UCB')

  ax1.xaxis.set_label_coords(0.5, -0.075)
  ax1.yaxis.set_label_coords(-0.1, 0.5)
  ax1.set_xlabel('p2', fontsize = 12, loc = 'center')
  ax1.set_ylabel('Regret', fontsize = 12, loc = 'center')
  ax1.legend()

  ax2.xaxis.set_ticks_position('bottom')
  ax2.yaxis.set_ticks_position('left')

  ax2.set_xlim(0, 0.9)
  ax2.set_ylim(-10, 200)  
  ax2.grid(True, which='both', axis='both', linestyle='--')

  ax2.plot(p2s, regrets_KL_UCB, 'red', label = 'KL-UCB')

  ax2.xaxis.set_label_coords(0.5, -0.075)
  ax2.yaxis.set_label_coords(-0.1, 0.5)
  ax2.set_xlabel('p2', fontsize = 12, loc = 'center')
  ax2.set_ylabel('Regret', fontsize = 12, loc = 'center')
  ax2.legend()

  fig.suptitle('Variation of Regret with p2 for KL-UCB and UCB algorithms')
  plt.savefig('regret_vs_p2_B.png',dpi=300)
  # plt.show()

