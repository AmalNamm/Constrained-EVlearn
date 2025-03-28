import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.sac import SAC as RLAgent
import time
import pandas as pd

dataset_name = 'cs5'
env = CityLearnEnv(dataset_name, central_agent=False)
averages = []
model = RLAgent(env)

start_time = time.time()
rewards, average_runtime, kpis_list = model.learn(episodes=10, keep_env_history=True, env_history_directory="./V2GENV_overnight28/sc2_sactestReward")
end_time = time.time()
elapsed_time = end_time - start_time

print("rewards")
print(rewards)

import numpy as np

kpis_list = np.array(kpis_list)  # If not already a NumPy array
np.save('./sc2_sac1/kpis_list.npy', kpis_list)

rewards = np.array(rewards)  # If not already a NumPy array
np.save('./sc2_sac1/rewards.npy', rewards)

print(elapsed_time)

kpis = model.env.evaluate().pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')
print(kpis)
