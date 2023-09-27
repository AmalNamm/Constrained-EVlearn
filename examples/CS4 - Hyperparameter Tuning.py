import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")

import seaborn as sns
import time
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpg import MADDPG as MADDPGAgent
from citylearn.reward_function import RewardFunction
from skopt.space import Real, Categorical
from skopt import gp_minimize
import pickle
from pandas.plotting import parallel_coordinates
from sklearn.manifold import TSNE

dataset_name = 'cs1'
env = CityLearnEnv(dataset_name, central_agent=False)
averages = []


def evaluate_hyperparameters(learning_rate_actor, learning_rate_critic, actor_units, critic_units, gamma, sigma,
                             steps_between_training_updates, target_update_interval, reward_type='cumulative'):
    model = MADDPGAgent(env, lr_actor=learning_rate_actor, lr_critic=learning_rate_critic,
                        actor_units=actor_units, critic_units=critic_units, gamma=gamma, sigma=sigma,
                        steps_between_training_updates=steps_between_training_updates,
                        target_update_interval=target_update_interval)

    start_time = time.time()
    rewards, average_runtime, kpis_list = model.learn(episodes=4)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Transforming the Data of rewards
    records = []
    for episode_idx, episode in enumerate(rewards):
        for timestep_idx, timestep in enumerate(episode):
            for building_idx, reward in enumerate(timestep):
                records.append((episode_idx, timestep_idx, building_idx, reward))
    df = pd.DataFrame(records, columns=["Episode", "Timestep", "Building", "Reward"])

    # Summing All Building Rewards Over an Episode
    sum_all_buildings = df.groupby("Episode").sum()["Reward"].reset_index()

    # Using the same reward type extraction logic
    if reward_type == 'cumulative':
        total_reward = sum(sum_all_buildings["Reward"])
    elif reward_type == 'last_episode':
        total_reward = sum_all_buildings.iloc[-1]["Reward"]
    else:
        raise ValueError("Invalid reward_type. Choose 'cumulative' or 'last_episode'")

    hyperparameters_dict = {
        'learning_rate_actor': learning_rate_actor,
        'learning_rate_critic': learning_rate_critic,
        'gamma': gamma,
        "actor_units": actor_units,
        "critic_units": critic_units,
        "steps_between_training_updates": steps_between_training_updates,
        "target_update_interval": target_update_interval,
        'sigma': sigma,
        'average_runtime': average_runtime,
        "runtime_training": elapsed_time,
        "total_reward": total_reward,
        'kpis_list': kpis_list
    }

    averages.append(hyperparameters_dict)
    return total_reward


current_generation = 0


# Update objective function
def objective(params):
    global current_generation
    current_generation += 1
    print(f"Generation {current_generation} of 50")
    time.sleep(10)  # Pauses the program for 5 seconds


    actor_units_key, critic_units_key, lr_actor, lr_critic, gamma, sigma, steps_between_training_updates, target_update_interval = params

    # Map units (if necessary or just pass the values directly)
    # Define mappings for actor and critic units
    critic_units_map = {
        '1024_512_256_128': [1024, 512, 256, 128],
        '1024_512_256': [1024, 512, 256],
        '512_256_128': [512, 256, 128],
        '256_128_64': [256, 128, 64]
    }
    actor_units_map = {
        '512_256_128': [512, 256, 128],
        '256_128_64': [256, 128, 64],
        '512_256': [512, 256],
        '256_128': [256, 128],
        '128_64': [128, 64]
    }

    actor_units = actor_units_map[actor_units_key]
    critic_units = critic_units_map[critic_units_key]

    rewards = evaluate_hyperparameters(actor_units=actor_units, critic_units=critic_units, learning_rate_actor=lr_actor,
                                       learning_rate_critic=lr_critic, gamma=gamma, sigma=sigma,
                                       steps_between_training_updates=steps_between_training_updates,
                                       target_update_interval=target_update_interval)
    return -np.mean(rewards)


# Define hyperparameter spaces
critic_units_space = Categorical(['1024_512_256_128', '1024_512_256', '512_256_128', '256_128_64'], name='actor_units')
actor_units_space = Categorical(['512_256_128', '256_128_64', '512_256', '256_128', '128_64'], name='critic_units')
lr_actor_space = Real(1e-5, 0.001, prior='log-uniform', name='lr_actor')
lr_critic_space = Real(1e-4, 0.001, prior='log-uniform', name='lr_critic')
gamma_space = Real(0.95, 0.99, name='gamma')
sigma_space = Real(0.2, 0.5, name='sigma')  # Assuming sigma values between 0.05 and 0.5
steps_between_training_updates_space = Categorical([10, 20, 50, 100], name='steps_between_training_updates')
target_update_interval_space = Categorical([20, 50, 100, 500],
                                           name='target_update_interval')  # Assuming some values for target update interval

# Combine all spaces into a list of dimensions
dimensions = [actor_units_space, critic_units_space, lr_actor_space, lr_critic_space, gamma_space,
              sigma_space, steps_between_training_updates_space, target_update_interval_space]

res = gp_minimize(objective, dimensions, n_calls=200, random_state=0)
# Saving the results of the Bayesian Optimization
with open('bayesian_optimization_result_V5.pkl', 'wb') as file:
    pickle.dump(res, file)

print("Best set of hyperparameters:", res.x)
print("Corresponding mean reward:", -res.fun)

# Adjusting the columns of the dataframe to reflect the hyperparameters in the updated search space
columns = ['Actor Units', 'Critic Units', 'LR Actor', 'LR Critic', 'Gamma', 'Sigma', 'Steps Between Training Updates',
           'Target Update Interval']
results_df = pd.DataFrame(res.x_iters, columns=columns)

# Adding the results and meta-data to the dataframe
results_df['Mean Reward'] = -res.func_vals
results_df['Experiment ID'] = range(1, len(res.x_iters) + 1)
results_df['Is Best'] = (results_df['Mean Reward'] == -res.fun)
results_df['Actor Units'] = results_df['Actor Units'].apply(lambda x: '_'.join(map(str, x)))
results_df['Critic Units'] = results_df['Critic Units'].apply(lambda x: '_'.join(map(str, x)))

# Displaying and saving the results
print(results_df)
results_df.to_csv('hyperparameter_optimization_results_V5.csv', index=False)

# Saving the average runtimes and other recorded metrics
average_runtime_df = pd.DataFrame(averages)
average_runtime_df.to_csv('average_runtime_results_V5.csv', index=False)

plt.plot(res.func_vals)
plt.title('Bayesian Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean Reward')
plt.grid(True)
plt.savefig('convergence_plotV5.png')
plt.show()

plt.scatter(results_df['LR Actor'], results_df['Mean Reward'])
plt.title('Mean Reward vs Learning Rate for Actor')
plt.xlabel('Learning Rate for Actor')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.savefig('mean_reward_vs_lr_actorV5.png')
plt.show()

plt.hist(results_df['Mean Reward'], bins=20)
plt.title('Distribution of Mean Rewards')
plt.xlabel('Mean Reward')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('distribution_mean_rewardsV5.png')
plt.show()

actor_units_mapping = {(256, 128): 1, (128, 64): 2}
results_df['Actor Units Mapped'] = results_df['Actor Units'].map(actor_units_mapping)

parallel_coordinates(
    results_df[['LR Actor', 'LR Critic', 'Gamma', 'Batch Size', 'Update Every', 'Actor Units Mapped', 'Mean Reward']],
    class_column='Mean Reward', color=('b', 'g', 'r', 'y', 'c', 'm'))
plt.title('Parallel Coordinates Plot of Hyperparameters')
plt.savefig('parallel_coordinates_plot.png')
plt.show()

results_df['Actor+Critic Units'] = results_df['Actor Units'].astype(str) + ' / ' + results_df['Critic Units'].astype(
    str)
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='Actor+Critic Units', y='Mean Reward')
plt.title('Distribution of Mean Rewards for Different Actor/Critic Units Configurations')
plt.savefig('boxplot_actor_critic_units.png')
plt.show()

sns.jointplot(data=results_df, x='LR Actor', y='Mean Reward', kind='scatter')
plt.title('Jointplot of Mean Reward vs Learning Rate for Actor')
plt.savefig('jointplot_mean_reward_vs_lr_actor.png')
plt.show()

X = results_df[['LR Actor', 'LR Critic', 'Gamma', 'Batch Size', 'Update Every', 'Actor Units Mapped']].values
y = results_df['Mean Reward'].values

X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
results_df['TSNE 1'] = X_tsne[:, 0]
results_df['TSNE 2'] = X_tsne[:, 1]

sns.scatterplot(data=results_df, x='TSNE 1', y='TSNE 2', hue='Mean Reward', palette='coolwarm')
plt.title('T-SNE Visualization of Hyperparameters and Mean Reward')
plt.savefig('tsne_visualization.png')
plt.show()
