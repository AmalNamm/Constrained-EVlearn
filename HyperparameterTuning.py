import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpgOptimized import MADDPGOptimizedRBC
from skopt.space import Real, Categorical
from skopt import gp_minimize
import pickle
from pandas.plotting import parallel_coordinates
from sklearn.manifold import TSNE
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F

class HyperparameterTuner:
    def __init__(self, env: CityLearnEnv, n_calls: int = 200, random_state: int = 0):
        self.env = env
        self.n_calls = n_calls
        self.random_state = random_state
        self.current_generation = 0
        self.averages = []
        
        # Create a more specific writer with timestamp
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f'runs/hyperparam_tuning/{current_time}')
        print(f"TensorBoard logs will be written to: runs/hyperparam_tuning/{current_time}")
        
        # Define mappings for actor and critic units
        self.critic_units_map = {
            '1024_512_256_128': [1024, 512, 256, 128],
            '1024_512_256': [1024, 512, 256],
            '512_256_128': [512, 256, 128],
            '256_128_64': [256, 128, 64]
        }
        self.actor_units_map = {
            '512_256_128': [512, 256, 128],
            '256_128_64': [256, 128, 64],
            '512_256': [512, 256],
            '256_128': [256, 128],
            '128_64': [128, 64]
        }

    def evaluate_hyperparameters(self, learning_rate_actor, learning_rate_critic, actor_units, critic_units, 
                               gamma, sigma, steps_between_training_updates, target_update_interval,
                               lr_dual, tau, decay_percentage, target_network):
        """Evaluate a set of hyperparameters by training the model and returning metrics."""
        print(f"\nEvaluating hyperparameters (Generation {self.current_generation}):")
        print(f"Actor units: {actor_units}")
        print(f"Critic units: {critic_units}")
        print(f"Learning rates - Actor: {learning_rate_actor:.2e}, Critic: {learning_rate_critic:.2e}, Dual: {lr_dual:.2e}")
        print(f"Gamma: {gamma:.3f}, Sigma: {sigma:.3f}, Tau: {tau:.3f}")
        print(f"Update intervals - Training: {steps_between_training_updates}, Target: {target_update_interval}")
        print(f"Target network: {target_network}, Decay: {decay_percentage:.3f}\n")

        model = MADDPGOptimizedRBC(self.env, 
                       lr_actor=learning_rate_actor, 
                       lr_critic=learning_rate_critic,
                       actor_units=actor_units, 
                       critic_units=critic_units, 
                       gamma=gamma, 
                       sigma=sigma,
                       steps_between_training_updates=steps_between_training_updates,
                       target_update_interval=target_update_interval,
                       lr_dual=lr_dual,
                       tau=tau,
                       decay_percentage=decay_percentage,
                       target_network=target_network)

        start_time = time.time()
        
        # Initialize lists to store metrics
        constraint_losses = []
        critic_losses = []
        actor_losses = []
        lambda_losses = []
        episode_rewards = []
        step_count = 0  # Global step counter for TensorBoard
        training_started = False
        
        # Run episodes
        for episode in range(4):
            print(f"\nEpisode {episode + 1}/4")
            episode_reward = 0
            observations = self.env.reset()
            done = False
            step = 0
            
            while not done:
                # Get actions from the model
                actions = model.predict(observations)
                
                # Take step in environment
                next_observations, rewards, done, _ = self.env.step(actions)  # type: ignore
                
                # Update the model and collect losses
                update_result = model.update(observations, actions, rewards, next_observations, done)
                
                if update_result is not None:
                    training_started = True
                    constraint_loss, critic_loss, lambda_loss = update_result
                    
                    # Only calculate actor loss and append losses if all components are available
                    if all(x is not None for x in [constraint_loss, critic_loss, lambda_loss]):
                        actor_loss = -float(critic_loss) + float(lambda_loss) * float(constraint_loss)
                        
                        constraint_losses.append(float(constraint_loss))
                        critic_losses.append(float(critic_loss))
                        lambda_losses.append(float(lambda_loss))
                        actor_losses.append(actor_loss)
                        
                        # Log individual steps to TensorBoard
                        self.writer.add_scalar('Training/Constraint_Loss', constraint_loss, step_count)
                        self.writer.add_scalar('Training/Critic_Loss', critic_loss, step_count)
                        self.writer.add_scalar('Training/Lambda_Loss', lambda_loss, step_count)
                        self.writer.add_scalar('Training/Actor_Loss', actor_loss, step_count)
                        self.writer.add_scalar('Training/Step_Reward', sum(rewards), step_count)
                        
                        # Print progress with losses every 100 steps
                        if step % 100 == 0:
                            print(f"Step {step}: Constraint Loss: {constraint_loss:.4f}, "
                                  f"Critic Loss: {critic_loss:.4f}, Lambda Loss: {lambda_loss:.4f}")
                elif not training_started and step % 100 == 0:
                    print(f"Step {step}: Filling replay buffer...")
                
                # Accumulate rewards
                episode_reward += sum(rewards)
                observations = next_observations
                step += 1
                step_count += 1
                
                # Print basic progress every 1000 steps
                if step % 1000 == 0:
                    print(f"Step {step}, Current episode reward: {episode_reward:.2f}")
                    
                # Flush TensorBoard writer periodically
                if step % 100 == 0:
                    self.writer.flush()
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished. Total reward: {episode_reward:.2f}")
            
            # Log episode metrics
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Steps', step, episode)
            self.writer.flush()  # Ensure episode data is written
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate average metrics
        avg_constraint_loss = np.mean(constraint_losses) if constraint_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_lambda_loss = np.mean(lambda_losses) if lambda_losses else 0
        total_reward = sum(episode_rewards)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total Runtime: {elapsed_time:.2f} seconds")
        print(f"Total Reward: {total_reward:.2f}")
        if training_started:
            print(f"Average Losses - Constraint: {avg_constraint_loss:.4f}, "
                  f"Critic: {avg_critic_loss:.4f}, Actor: {avg_actor_loss:.4f}, "
                  f"Lambda: {avg_lambda_loss:.4f}")
        else:
            print("Training did not start - replay buffer not filled")
        print()
        
        # Log hyperparameters and final metrics
        hparams = {
            'lr_actor': learning_rate_actor,
            'lr_critic': learning_rate_critic,
            'lr_dual': lr_dual,
            'gamma': gamma,
            'sigma': sigma,
            'tau': tau,
            'decay_percentage': decay_percentage,
            'target_network': target_network,
            'steps_between_updates': steps_between_training_updates,
            'target_update_interval': target_update_interval,
        }
        
        metrics = {
            'hparam/avg_reward': total_reward / 4,
            'hparam/avg_constraint_loss': avg_constraint_loss,
            'hparam/avg_critic_loss': avg_critic_loss,
            'hparam/avg_actor_loss': avg_actor_loss,
            'hparam/avg_lambda_loss': avg_lambda_loss,
            'hparam/runtime': elapsed_time
        }
        
        self.writer.add_hparams(hparams, metrics)
        self.writer.flush()  # Ensure all data is written

        hyperparameters_dict = {
            'learning_rate_actor': learning_rate_actor,
            'learning_rate_critic': learning_rate_critic,
            'gamma': gamma,
            "actor_units": actor_units,
            "critic_units": critic_units,
            "steps_between_training_updates": steps_between_training_updates,
            "target_update_interval": target_update_interval,
            'sigma': sigma,
            'lr_dual': lr_dual,
            'tau': tau,
            'decay_percentage': decay_percentage,
            'target_network': target_network,
            "runtime_training": elapsed_time,
            "total_reward": total_reward,
            'avg_constraint_loss': avg_constraint_loss,
            'avg_critic_loss': avg_critic_loss,
            'avg_actor_loss': avg_actor_loss,
            'avg_lambda_loss': avg_lambda_loss
        }

        self.averages.append(hyperparameters_dict)
        return total_reward, avg_constraint_loss, avg_critic_loss, avg_actor_loss, avg_lambda_loss

    def objective(self, params):
        """Objective function for Bayesian optimization."""
        self.current_generation += 1
        print(f"Generation {self.current_generation} of {self.n_calls}")
        
        (actor_units_key, critic_units_key, lr_actor, lr_critic, gamma, sigma, 
         steps_between_training_updates, target_update_interval, lr_dual, tau, 
         decay_percentage, target_network) = params
        
        actor_units = self.actor_units_map[actor_units_key]
        critic_units = self.critic_units_map[critic_units_key]
        
        total_reward, constraint_loss, critic_loss, actor_loss, lambda_loss = self.evaluate_hyperparameters(
            actor_units=actor_units, 
            critic_units=critic_units, 
            learning_rate_actor=lr_actor,
            learning_rate_critic=lr_critic, 
            gamma=gamma, 
            sigma=sigma,
            steps_between_training_updates=steps_between_training_updates,
            target_update_interval=target_update_interval,
            lr_dual=lr_dual,
            tau=tau,
            decay_percentage=decay_percentage,
            target_network=target_network
        )
        
        # Weighted sum of objectives
        weighted_loss = -total_reward + 0.1 * constraint_loss + 0.1 * critic_loss + 0.1 * actor_loss + 0.1 * lambda_loss
        return weighted_loss

    def optimize(self):
        """Run the hyperparameter optimization."""
        # Define hyperparameter spaces
        critic_units_space = Categorical(['1024_512_256_128', '1024_512_256', '512_256_128', '256_128_64'], 
                                       name='critic_units')
        actor_units_space = Categorical(['512_256_128', '256_128_64', '512_256', '256_128', '128_64'], 
                                      name='actor_units')
        lr_actor_space = Real(1e-5, 0.001, prior='log-uniform', name='lr_actor')
        lr_critic_space = Real(1e-4, 0.001, prior='log-uniform', name='lr_critic')
        gamma_space = Real(0.95, 0.99, name='gamma')
        sigma_space = Real(0.2, 0.5, name='sigma')
        steps_between_training_updates_space = Categorical([10, 20, 50, 100], 
                                                         name='steps_between_training_updates')
        target_update_interval_space = Categorical([20, 50, 100, 500], 
                                                 name='target_update_interval')
        lr_dual_space = Real(1e-6, 1e-4, prior='log-uniform', name='lr_dual')
        tau_space = Real(1e-4, 1e-2, prior='log-uniform', name='tau')
        decay_percentage_space = Real(0.99, 0.999, name='decay_percentage')
        target_network_space = Categorical([True, False], name='target_network')

        # Combine all spaces
        dimensions = [
            actor_units_space, critic_units_space, lr_actor_space, lr_critic_space, 
            gamma_space, sigma_space, steps_between_training_updates_space, 
            target_update_interval_space, lr_dual_space, tau_space, 
            decay_percentage_space, target_network_space
        ]

        # Run optimization
        res = gp_minimize(self.objective, dimensions, n_calls=self.n_calls, 
                         random_state=self.random_state)

        # Save results
        self.save_results(res)
        return res

    def save_results(self, res):
        """Save optimization results and generate plots."""
        # Save optimization results
        with open('bayesian_optimization_result.pkl', 'wb') as file:
            pickle.dump(res, file)

        # Create results DataFrame
        results_df = pd.DataFrame(self.averages)
        results_df.to_csv('hyperparameter_optimization_results.csv', index=False)

        # Generate plots
        self.plot_results(results_df)

    def plot_results(self, results_df):
        """Generate and save various analysis plots."""
        # Loss convergence plots
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(results_df.index, results_df['avg_constraint_loss'])
        plt.title('Constraint Loss Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Constraint Loss')

        plt.subplot(2, 2, 2)
        plt.plot(results_df.index, results_df['avg_critic_loss'])
        plt.title('Critic Loss Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Critic Loss')

        plt.subplot(2, 2, 3)
        plt.plot(results_df.index, results_df['avg_actor_loss'])
        plt.title('Actor Loss Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Actor Loss')

        plt.subplot(2, 2, 4)
        plt.plot(results_df.index, results_df['avg_lambda_loss'])
        plt.title('Lambda Loss Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Lambda Loss')

        plt.tight_layout()
        plt.savefig('loss_convergence_plots.png')
        plt.close()

        # Loss correlations
        plt.figure(figsize=(15, 10))
        sns.heatmap(results_df[['avg_constraint_loss', 'avg_critic_loss', 'avg_actor_loss', 
                               'avg_lambda_loss', 'total_reward']].corr(), 
                    annot=True, cmap='coolwarm')
        plt.title('Loss Correlations')
        plt.savefig('loss_correlations.png')
        plt.close()

        # Parallel coordinates plot
        plt.figure(figsize=(15, 10))
        parallel_coordinates(
            results_df[['learning_rate_actor', 'learning_rate_critic', 'gamma', 
                       'steps_between_training_updates', 'target_update_interval', 
                       'total_reward']],
            class_column='total_reward')
        plt.title('Parallel Coordinates Plot of Hyperparameters')
        plt.savefig('parallel_coordinates_plot.png')
        plt.close()

        # T-SNE visualization
        X = results_df[['learning_rate_actor', 'learning_rate_critic', 'gamma', 
                       'steps_between_training_updates', 'target_update_interval']].values
        X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=results_df['total_reward'], 
                   cmap='viridis')
        plt.colorbar(label='Total Reward')
        plt.title('T-SNE Visualization of Hyperparameters')
        plt.savefig('tsne_visualization.png')
        plt.close()

def main():
    # Initialize environment
    #dataset_name = 'cs1'  # or your preferred dataset
    dataset_name = "/home/amalnamm/work/CMDPs/Constrained-EVlearn/data-test/Tiago_dataset/Tiagoschema.json"
    env = CityLearnEnv(dataset_name, central_agent=False)
    
    # Initialize and run hyperparameter tuning
    tuner = HyperparameterTuner(env, n_calls=200, random_state=0)
    results = tuner.optimize()
    
    print("Best set of hyperparameters:", results.x)
    print("Corresponding mean reward:", -results.fun)

if __name__ == "__main__":
    main() 