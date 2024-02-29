import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import os
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")
from citylearn.reward_function import V2GPenaltyReward
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpg import MADDPG as RLAgent
import time
import torch

dataset_name = 'cs5'
#dataset_name = 'cs1'
episodes = 10
path_save = "./tiago_thesis/Empirical_Reward_Test/MADDPGRBC_scaled_rewards_v1_without_squash_all_D_V6_NewRBCCOMPARISON"
path_save = "./tiago_thesis/TESTTT"
#
env = CityLearnEnv(dataset_name, central_agent=False, reward_function=V2GPenaltyReward)
##env.reward_function.SQUASH = 1
#
#
#averages = []
##model = RLAgent(env)
model = RLAgent(env, critic_units=[512, 256, 128], actor_units=[256, 128, 64], lr_actor=0.0006343946342268605, lr_critic=0.0009067117952187151, gamma=0.9773507798877807, sigma=0.2264587893937525, steps_between_training_updates=20, target_update_interval=100)
start_time = time.time()
rewards, average_runtime, kpis_list, observation_ep  = model.learn(episodes=episodes)
end_time = time.time()
elapsed_time = end_time - start_time
#
#
#
## 1. Transforming the Data of rewards
#records = []
#for episode_idx, episode in enumerate(rewards):
#    for timestep_idx, timestep in enumerate(episode):
#        for building_idx, reward in enumerate(timestep):
#            records.append((episode_idx, timestep_idx, building_idx, reward))
#df = pd.DataFrame(records, columns=["Episode", "Timestep", "Building", "Reward"])
#
## Save the main dataframe
#name_of_file = '/main_rewards_data.csv'
#df.to_csv(path_save + name_of_file, index=False)
#
## 2. Fetching Timeseries for a Building (for building 0 and episode 0 in this example)
#timeseries_building0_ep0 = df[(df['Building'] == 0) & (df['Episode'] == 0)][["Timestep", "Reward"]]
#timeseries_building0_ep0.to_csv(path_save + '/timeseries_building0_ep0.csv', index=False)
#
## 3. Summing Rewards for a Building Over an Episode
#sum_rewards_per_building = df.groupby(["Episode", "Building"]).sum()["Reward"].reset_index()
#sum_rewards_per_building.to_csv(path_save + '/sum_rewards_per_building.csv', index=False)
#print(sum_rewards_per_building)
#
## 4. Summing All Building Rewards Over an Episode
#sum_all_buildings = df.groupby("Episode").sum()["Reward"].reset_index()
#sum_all_buildings.to_csv(path_save + '/sum_all_buildings.csv', index=False)
#print(sum_all_buildings)
#
#kpis_dir = os.path.join(path_save, 'kpis_list')
#
## Create the directory if it doesn't exist
#if not os.path.exists(kpis_dir):
#    os.makedirs(kpis_dir)
#
#for idx, df in enumerate(kpis_list, 1):
#    df.to_csv(os.path.join(kpis_dir, f'df_{idx}.csv'), index=False)
#
#text = f"\nAverage prediction time over {episodes} episodes: {average_runtime}\n" \
#      f"ATotal training time over {episodes} episodes: {elapsed_time}\n"
#print(text)
#
## Saving the elapsed_time variable
#with open(path_save + '/elapsed_time.pkl', 'wb') as f:
#    pickle.dump(text, f)
#
#kpis = model.env.evaluate().pivot(index='cost_function', columns='name', values='value')
#kpis = kpis.dropna(how='all')
#print(kpis)
#
#
## Save the model
#model.save_maddpg_model("maddpg_trained.pth")

# Later on, or in another script:
# agent = MADDPG(env)
# Load the model
#loaded_agent = RLAgent.from_saved_model("maddpg_trained.pth")

#def load_from_pickle(file_name):
#    data = []
#    with open(file_name, 'rb') as f:
#        while True:
#            try:
#                data.append(pickle.load(f))
#            except EOFError:  # This error will be raised when we reach the end of the file.
#                break
#    return data
#
#stored_data = load_from_pickle('method_calls.pkl')
#individual_runtimes_predict = []
#for timestep_array in stored_data:
#    print(timestep_array)
#    start_time = time.time()  # Get the current time
#    actions = loaded_agent.predict_deterministic(timestep_array)
#    end_time = time.time()  # Get the current time again after the function has run
#    elapsed_time = end_time - start_time  # Calculate the elapsed time
#    individual_runtimes_predict.append(elapsed_time)
#print(sum(individual_runtimes_predict) / len(individual_runtimes_predict))

#with open('method_calls.pkl', 'rb') as f:
#    try:
#        while True:
#            data = pickle.load(f)
#            individual_runtimes_predict = []
#            for observations in data:
#                start_time = time.time()  # Get the current time
#                actions = loaded_agent.predict_deterministic(observations)
#                end_time = time.time()  # Get the current time again after the function has run
#                elapsed_time = end_time - start_time  # Calculate the elapsed time
#                individual_runtimes_predict.append(elapsed_time)
#
#            print(sum(individual_runtimes_predict) / len(individual_runtimes_predict))
#    except EOFError:
#        pass



