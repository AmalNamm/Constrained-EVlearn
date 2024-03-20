import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpg import MADDPGOptimizedRBC as RLAgent
from citylearn.reward_function import V2GPenaltyReward

import time

dataset_name = 'cs5'
env = CityLearnEnv(dataset_name, central_agent=False, reward_function=V2GPenaltyReward)
averages = []
model = RLAgent(env, critic_units=[512, 256, 128], actor_units=[256, 128, 64], lr_actor=0.0006343946342268605, lr_critic=0.0009067117952187151, gamma=0.9773507798877807, sigma=0.2264587893937525, steps_between_training_updates=20, target_update_interval=100)

start_time = time.time()
#rewards, average_runtime, kpis_list = model.learn(episodes=10, keep_env_history=True, env_history_directory="./V2GENV_overnight28/sc2_maddpgrbctestReward")
rewards, average_runtime, kpis_list = model.learn(episodes=20, env_history_directory="./V2GENV/INESCTECTest1")
end_time = time.time()
elapsed_time = end_time - start_time

print("rewards")
print(rewards)

import numpy as np

kpis_list = np.array(kpis_list)  # If not already a NumPy array
np.save('./sc2_maddpgrbctestReward/kpis_list.npy', kpis_list)

rewards = np.array(rewards)  # If not already a NumPy array
np.save('./sc2_maddpgrbctestReward/rewards.npy', rewards)

print(elapsed_time)

kpis = model.env.evaluate().pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')
print(kpis)
