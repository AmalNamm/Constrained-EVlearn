import os
import argparse
import time
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpgOptimized import MADDPGOptimizedRBC as RLAgent
from citylearn.reward_function import SimpleEVReward

#python Train_CMaddpg_multi_reward.py --reward_function SimpleEVReward --reward_scaling 0.001
#tensorboard --logdir=./logs_Tensorboard
#python Train_Simple_Reward.py --reward_function SimpleEVReward --reward_scaling 0.001
#  lr_dual ≥ lr_critic ≥ lr_actor
def main(schema_path: str, log_dir: str, episodes: int, reward_function_name: str, reward_scaling: float):
    # Create log directory with reward function name and scaling
    reward_log_dir = os.path.join(log_dir, f"{reward_function_name}_scale{reward_scaling}")
    os.makedirs(reward_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=reward_log_dir)
    print(f"Logging to TensorBoard at: {reward_log_dir}")
    print(f"Using reward scaling: {reward_scaling}")

    # Import and select reward function based on argument
    if reward_function_name == "V2GPenaltyReward":
        from citylearn.reward_function import V2GPenaltyReward
        reward_function = V2GPenaltyReward
    elif reward_function_name == "MARL":
        from citylearn.reward_function import MARL
        reward_function = MARL
    elif reward_function_name == "IndependentSACReward":
        from citylearn.reward_function import IndependentSACReward
        reward_function = IndependentSACReward
    elif reward_function_name == "SolarPenaltyReward":
        from citylearn.reward_function import SolarPenaltyReward
        reward_function = SolarPenaltyReward
    elif reward_function_name == "SimpleEVReward":
        # Already imported at the top - pass scaling parameter
        # We'll create the instantiated reward object later
        reward_class = SimpleEVReward
        reward_function = lambda env: reward_class(env, reward_scaling=reward_scaling)
    else:
        from citylearn.reward_function import V2GPenaltyReward
        reward_function = V2GPenaltyReward
        print(f"Unknown reward function: {reward_function_name}, using V2GPenaltyReward instead")

    env = CityLearnEnv(schema=schema_path, central_agent=False, reward_function=reward_function)

    # Define agent parameters
    gamma = 0.9773507798877807  # Discount factor
    
    # Calculate TGeLU range based on reward scale and discount factor
    max_reward_estimate = 10.0 / reward_scaling if reward_function_name == "SimpleEVReward" else 10.0
    safety_factor = 2.0
    tgelu_bound = safety_factor * max_reward_estimate / (1 - gamma)
    
    print(f"Discount factor (gamma): {gamma}")
    print(f"Estimated max reward: {max_reward_estimate}")
    print(f"Calculated TGeLU range: [{-tgelu_bound:.2f}, {tgelu_bound:.2f}]")
    print(f"Note: Actor networks will use a dynamic TGeLU range calculated from these values.")

    
    model = RLAgent(
        env,
        critic_units=[512, 256, 128],
        actor_units=[256, 128, 64],
        lr_actor=1e-4,
        lr_critic=1e-4,
        #gamma=0.9773507798877807,
        gamma=gamma,
        sigma=0.2264587893937525,
        lr_dual=1e-3,
        steps_between_training_updates=20,
        target_update_interval=100
    )

    # Add callback for monitoring reward vs constraint scales
    def monitor_reward_constraint_scale(rewards, constraints, step):
        if step % 100 == 0 and step > 0:
            # Calculate statistics for rewards and constraints
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            avg_constraint = np.mean(constraints[-100:]) if len(constraints) >= 100 else np.mean(constraints)
            
            # Log the ratio to check scale balance
            ratio = abs(avg_reward / (avg_constraint + 1e-10))
            
            # Log to TensorBoard
            writer.add_scalar('Scale/Reward_Avg', avg_reward, step)
            writer.add_scalar('Scale/Constraint_Avg', avg_constraint, step)
            writer.add_scalar('Scale/Reward_Constraint_Ratio', ratio, step)
            
            # Print the values for monitoring
            print(f"Step {step}: Avg Reward = {avg_reward:.6f}, Avg Constraint = {avg_constraint:.6f}, Ratio = {ratio:.2f}")
            
            # Ideal ratio would be around 1.0, meaning scales are balanced
            if ratio > 10 or ratio < 0.1:
                print(f"WARNING: Reward and constraint scales are imbalanced (ratio = {ratio:.2f})")
                print(f"Consider adjusting reward_scaling to bring this ratio closer to 1.0")
    
    # Add the callback to the model
    model.scale_monitor_callback = monitor_reward_constraint_scale

    rewards_all, avg_runtime, kpis, obs, constraints, q_vals, lambdas = model.learn(
        episodes=episodes,
        deterministic=False,
        deterministic_finish=False,
        keep_env_history=True,
        writer=writer
    )

    writer.close()
    print(f"Training complete for {reward_function_name}. Use `tensorboard --logdir {reward_log_dir}` to visualize results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema_path", 
        type=str,
        default="/home/amalnamm/work/CMDPs/Constrained-EVlearn/data-test/Tiago_dataset/Tiagoschema.json",
        help="Path to the schema.json"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs_Tensorboard", 
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=40, 
        help="Number of training episodes"
    )
    # Add reward function argument with SimpleEVReward added to choices
    parser.add_argument(
        "--reward_function", 
        type=str, 
        default="SimpleEVReward", 
        choices=["V2GPenaltyReward", "MARL", "IndependentSACReward", "SolarPenaltyReward", "SimpleEVReward"],
        help="Reward function to use"
    )
    # Add reward scaling argument
    parser.add_argument(
        "--reward_scaling",
        type=float,
        default=0.001,
        help="Scaling factor for rewards to match constraint scale"
    )

    args = parser.parse_args()
    main(args.schema_path, args.log_dir, args.episodes, args.reward_function, args.reward_scaling) 