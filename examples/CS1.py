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

dataset_name = 'tiago_thesis_test_1'
env = CityLearnEnv(dataset_name, central_agent=False, reward_function=RewardFunction)
averages = []

def evaluate_hyperparameters(learning_rate_actor, learning_rate_critic, batch_size, gamma, actor_units, critic_units, update_every, reward_type='cumulative'):
    model = MADDPGAgent(env, lr_actor=learning_rate_actor, lr_critic=learning_rate_critic,
                        batch_size=batch_size, gamma=gamma, actor_units=actor_units, critic_units=critic_units, update_every=update_every)

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
        'learning_rate_actor': learning_rate_actor,
        'learning_rate_critic': learning_rate_critic,
        'batch_size': batch_size,
        'gamma': gamma,
        "actor_units": actor_units,
        "critic_units": critic_units,
        "update_every": update_every,
        'average_runtime': average_runtime,
        "runtime_training": elapsed_time,
        "total_reward": total_reward,
        'kpis_list': kpis_list
    }

    averages.append(hyperparameters_dict)
    return total_reward

current_generation = 0

def objective(params):
    global current_generation
    current_generation += 1
    print(f"Generation {current_generation} of 50")
    actor_units_map = {'512_256': (256, 128), '256_128': (256, 128), '128_64': (128, 64)}
    critic_units_map = {'512_256': (256, 128), '256_128': (256, 128), '128_64': (128, 64)}

    actor_units_key, critic_units_key, lr_actor, lr_critic, gamma, batch_size, update_every_key = params
    actor_units = actor_units_map[actor_units_key]
    critic_units = critic_units_map[critic_units_key]

    rewards = evaluate_hyperparameters(actor_units=actor_units, critic_units=critic_units, learning_rate_actor=lr_actor,
                                       learning_rate_critic=lr_critic, gamma=gamma, batch_size=batch_size,
                                       update_every=update_every_key)
    return -np.mean(rewards)

actor_units_space = Categorical(['512_256','256_128', '128_64'], name='actor_units')
critic_units_space = Categorical(['512_256', '256_128', '128_64'], name='critic_units')
lr_actor_space = Real(1e-4, 0.01, prior='log-uniform', name='lr_actor')
lr_critic_space = Real(1e-4, 0.01, prior='log-uniform', name='lr_critic')
gamma_space = Real(0.95, 0.99, name='gamma')
batch_size_space = Categorical([64, 128, 256], name='batch_size')
update_every = Categorical([1, 5, 10, 20], name='update_every')

dimensions = [actor_units_space, critic_units_space, lr_actor_space, lr_critic_space, gamma_space, batch_size_space, update_every]

res = gp_minimize(objective, dimensions, n_calls=50, random_state=0)

with open('bayesian_optimization_result.pkl', 'wb') as file:
    pickle.dump(res, file)

print("Best set of hyperparameters:", res.x)
print("Corresponding mean reward:", -res.fun)

results_df = pd.DataFrame(res.x_iters, columns=['Actor Units', 'Critic Units', 'LR Actor', 'LR Critic', 'Gamma', 'Batch Size', 'Update Every'])
results_df['Mean Reward'] = -res.func_vals
results_df['Experiment ID'] = range(1, len(res.x_iters) + 1)
results_df['Is Best'] = (results_df['Mean Reward'] == -res.fun)
print(results_df)

results_df.to_csv('hyperparameter_optimization_results.csv', index=False)

average_runtime_df = pd.DataFrame(averages)
average_runtime_df.to_csv('average_runtime_results.csv', index=False)

plt.plot(res.func_vals)
plt.title('Bayesian Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean Reward')
plt.grid(True)
plt.savefig('convergence_plot.png')
plt.show()

plt.scatter(results_df['LR Actor'], results_df['Mean Reward'])
plt.title('Mean Reward vs Learning Rate for Actor')
plt.xlabel('Learning Rate for Actor')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.savefig('mean_reward_vs_lr_actor.png')
plt.show()

plt.hist(results_df['Mean Reward'], bins=20)
plt.title('Distribution of Mean Rewards')
plt.xlabel('Mean Reward')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('distribution_mean_rewards.png')
plt.show()

actor_units_mapping = {(256, 128): 1, (128, 64): 2}
results_df['Actor Units Mapped'] = results_df['Actor Units'].map(actor_units_mapping)

parallel_coordinates(results_df[['LR Actor', 'LR Critic', 'Gamma', 'Batch Size', 'Update Every', 'Actor Units Mapped', 'Mean Reward']], class_column='Mean Reward', color=('b', 'g', 'r', 'y', 'c', 'm'))
plt.title('Parallel Coordinates Plot of Hyperparameters')
plt.savefig('parallel_coordinates_plot.png')
plt.show()

results_df['Actor+Critic Units'] = results_df['Actor Units'].astype(str) + ' / ' + results_df['Critic Units'].astype(str)
plt.figure(figsize=(10,6))
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
results_df['TSNE 1'] = X_tsne[:,0]
results_df['TSNE 2'] = X_tsne[:,1]

sns.scatterplot(data=results_df, x='TSNE 1', y='TSNE 2', hue='Mean Reward', palette='coolwarm')
plt.title('T-SNE Visualization of Hyperparameters and Mean Reward')
plt.savefig('tsne_visualization.png')
plt.show()
