import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")

import seaborn as sns
import time
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpg import MADDPG as MADDPGAgent
from citylearn.reward_function import V2GPenaltyReward
from skopt.space import Real, Categorical
from skopt import gp_minimize
import pickle
from pandas.plotting import parallel_coordinates
from sklearn.manifold import TSNE

dataset_name = 'cs5'
averages = []
current_generation = 0

def evaluate_hyperparameters(peak_percentage_threshold, ramping_percentage_threshold, peak_penalty_weight,
                            ramping_penalty_weight, energy_transfer_bonus, window_size, penalty_no_car_charging,
                            reward_type='last_episode'):


    env = CityLearnEnv(dataset_name, central_agent=False, reward_function=V2GPenaltyReward(env,
                                    peak_percentage_threshold=peak_percentage_threshold,
                                    ramping_percentage_threshold=ramping_percentage_threshold,
                                    peak_penalty_weight=peak_penalty_weight,
                                    ramping_penalty_weight=ramping_penalty_weight,
                                    energy_transfer_bonus=energy_transfer_bonus,
                                    window_size=window_size,
                                    penalty_no_car_charging=penalty_no_car_charging))

    model = MADDPGAgent(env, episodes=6, keep_env_history=True, env_history_directory=f"./sc2_maddpgrbctestReward_{current_generation}")

    start_time = time.time()
    rewards, average_runtime, kpis_list = model.learn(episodes=6)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if reward_type == 'cumulative':
        total_reward = sum(rewards)
    elif reward_type == 'last_episode':
        total_reward = rewards[-1]
    else:
        raise ValueError("Invalid reward_type. Choose 'cumulative' or 'last_episode'")

    hyperparameters_dict = {
        'peak_percentage_threshold': peak_percentage_threshold,
        'ramping_percentage_threshold': ramping_percentage_threshold,
        'peak_penalty_weight': peak_penalty_weight,
        'ramping_penalty_weight': ramping_penalty_weight,
        'energy_transfer_bonus': energy_transfer_bonus,
        'window_size': window_size,
        'penalty_no_car_charging': penalty_no_car_charging,
        'average_runtime': average_runtime,
        "runtime_training": elapsed_time,
        "total_reward": total_reward,
        'kpis_list': kpis_list
    }

    averages.append(hyperparameters_dict)
    return total_reward



def objective(params):
    global current_generation
    current_generation += 1
    print(f"Generation {current_generation} of 50")
    peak_percentage_threshold, ramping_percentage_threshold, peak_penalty_weight, ramping_penalty_weight, energy_transfer_bonus, window_size, penalty_no_car_charging = params

    rewards = evaluate_hyperparameters(peak_percentage_threshold, ramping_percentage_threshold, peak_penalty_weight,
                                       ramping_penalty_weight, energy_transfer_bonus, window_size,
                                       penalty_no_car_charging)
    return -np.mean(rewards)

# Parameter spaces for V2GPenaltyReward
peak_percentage_threshold_space = Real(0.01, 0.5, name='peak_percentage_threshold')
ramping_percentage_threshold_space = Real(0.01, 0.5, name='ramping_percentage_threshold')
peak_penalty_weight_space = Real(1, 5, name='peak_penalty_weight')
ramping_penalty_weight_space = Real(1, 5, name='ramping_penalty_weight')
energy_transfer_bonus_space = Real(0.1, 2, name='energy_transfer_bonus')
window_size_space = Integer(3, 10, name='window_size')
penalty_no_car_charging_space = Integer(-200, -10, name='penalty_no_car_charging')

dimensions = [peak_percentage_threshold_space, ramping_percentage_threshold_space, peak_penalty_weight_space,
             ramping_penalty_weight_space, energy_transfer_bonus_space, window_size_space, penalty_no_car_charging_space]

res = gp_minimize(objective, dimensions, n_calls=50, random_state=0)

with open('bayesian_optimization_result.pkl', 'wb') as file:
    pickle.dump(res, file)

print("Best set of parameters:", res.x)
print("Corresponding mean reward:", -res.fun)

# Create DataFrame to hold the results
columns = ['Peak Percentage Threshold', 'Ramping Percentage Threshold', 'Peak Penalty Weight',
           'Ramping Penalty Weight', 'Energy Transfer Bonus', 'Window Size', 'Penalty No Car Charging', 'Mean Reward']
results_df = pd.DataFrame(res.x_iters, columns=columns)
results_df['Mean Reward'] = -res.func_vals
results_df['Experiment ID'] = range(1, len(res.x_iters) + 1)
results_df['Is Best'] = (results_df['Mean Reward'] == -res.fun)
print(results_df)

results_df.to_csv('1reward_hyperparameter_optimization_results.csv', index=False)

# Save the detailed averages data
average_runtime_df = pd.DataFrame(averages)
average_runtime_df.to_csv('1reward_average_runtime_results.csv', index=False)

# Bayesian Optimization Progress
plt.plot(res.func_vals)
plt.title('Bayesian Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean Reward')
plt.grid(True)
plt.savefig('convergence_plot.png')
plt.show()

# Distribution of Mean Rewards
plt.hist(results_df['Mean Reward'], bins=20)
plt.title('Distribution of Mean Rewards')
plt.xlabel('Mean Reward')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('distribution_mean_rewards.png')
plt.show()

# Note: Since the previous plots you've provided as reference are based on MADDPG hyperparameters (like 'LR Actor', 'Actor Units', etc.),
# I've removed those plots. However, if you need visualization related to `V2GPenaltyReward` parameters like `peak_percentage_threshold`,
# let me know and I can assist further.

# T-SNE Visualization of Hyperparameters and Mean Reward
X = results_df[['Peak Percentage Threshold', 'Ramping Percentage Threshold', 'Peak Penalty Weight',
                'Ramping Penalty Weight', 'Energy Transfer Bonus', 'Window Size', 'Penalty No Car Charging']].values
y = results_df['Mean Reward'].values

X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
results_df['TSNE 1'] = X_tsne[:,0]
results_df['TSNE 2'] = X_tsne[:,1]

sns.scatterplot(data=results_df, x='TSNE 1', y='TSNE 2', hue='Mean Reward', palette='coolwarm')
plt.title('T-SNE Visualization of Hyperparameters and Mean Reward')
plt.savefig('tsne_visualization.png')
plt.show()

