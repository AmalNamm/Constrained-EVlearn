import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpg import MADDPGOptimizedRBC as RLAgent
import time

dataset_name = 'cs1'
env = CityLearnEnv(dataset_name, central_agent=False)
averages = []
model = RLAgent(env, update_every=10)

start_time = time.time()
rewards, average_runtime, kpis_list = model.learn(episodes=6, keep_env_history=True, env_history_directory="./sc2_maddpgrbcsolar17")
end_time = time.time()
elapsed_time = end_time - start_time

print("rewards")
print(rewards)

import numpy as np

kpis_list = np.array(kpis_list)  # If not already a NumPy array
np.save('./sc2_maddpgrbcsolar17/kpis_list.npy', kpis_list)

rewards = np.array(rewards)  # If not already a NumPy array
np.save('./sc2_maddpgrbcsolar17/rewards.npy', rewards)

print(elapsed_time)

kpis = model.env.evaluate().pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')
print(kpis)
