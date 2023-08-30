import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import os
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")
from citylearn.reward_function import V2GPenaltyReward
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import V2GRBC as RLAgent
import time

dataset_name = 'cs5'
episodes = 1
path_save = "./tiago_thesis/Empirical_Reward_Test/RBC"

env = CityLearnEnv(dataset_name, central_agent=False, reward_function=V2GPenaltyReward)


averages = []
model = RLAgent(env)

start_time = time.time()
rewards, average_runtime, kpis_list = model.learn(episodes=episodes,
                                                  keep_env_history=True,
                                                  env_history_directory=path_save)
end_time = time.time()
elapsed_time = end_time - start_time



# 1. Transforming the Data of rewards
records = []
for episode_idx, episode in enumerate(rewards):
    for timestep_idx, timestep in enumerate(episode):
        for building_idx, reward in enumerate(timestep):
            records.append((episode_idx, timestep_idx, building_idx, reward))
df = pd.DataFrame(records, columns=["Episode", "Timestep", "Building", "Reward"])

# Save the main dataframe
name_of_file = '/main_rewards_data.csv'
df.to_csv(path_save + name_of_file, index=False)

# 2. Fetching Timeseries for a Building (for building 0 and episode 0 in this example)
timeseries_building0_ep0 = df[(df['Building'] == 0) & (df['Episode'] == 0)][["Timestep", "Reward"]]
timeseries_building0_ep0.to_csv(path_save + '/timeseries_building0_ep0.csv', index=False)

# 3. Summing Rewards for a Building Over an Episode
sum_rewards_per_building = df.groupby(["Episode", "Building"]).sum()["Reward"].reset_index()
sum_rewards_per_building.to_csv(path_save + '/sum_rewards_per_building.csv', index=False)
print(sum_rewards_per_building)

# 4. Summing All Building Rewards Over an Episode
sum_all_buildings = df.groupby("Episode").sum()["Reward"].reset_index()
sum_all_buildings.to_csv(path_save + '/sum_all_buildings.csv', index=False)
print(sum_all_buildings)

kpis_dir = os.path.join(path_save, 'kpis_list')

# Create the directory if it doesn't exist
if not os.path.exists(kpis_dir):
    os.makedirs(kpis_dir)

for idx, df in enumerate(kpis_list, 1):
    df.to_csv(os.path.join(kpis_dir, f'df_{idx}.csv'), index=False)

text = f"\nAverage prediction time over {episodes} episodes: {average_runtime}\n" \
      f"ATotal training time over {episodes} episodes: {elapsed_time}\n"
print(text)

# Saving the elapsed_time variable
with open(path_save + '/elapsed_time.pkl', 'wb') as f:
    pickle.dump(text, f)

kpis = model.env.evaluate().pivot(index='cost_function', columns='name', values='value')
kpis = kpis.dropna(how='all')
print(kpis)
