import os
import argparse
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpgOptimized import MADDPGOptimizedRBC as RLAgent
from citylearn.reward_function import V2GPenaltyReward, IndependentSACReward, MARL, SolarPenaltyReward
#python Train_CMaddpg_multi_reward.py --reward_function IndependentSACReward
#tensorboard --logdir=./logs_Tensorboard

#python Train_multi_reward.py --reward_function V2GPenaltyReward
#python Train_multi_reward.py --reward_function IndependentSACReward
#python Train_multi_reward.py --reward_function MARL
#python Train_multi_reward.py --reward_function SolarPenaltyReward

#  lr_dual ≥ lr_critic ≥ lr_actor
def main(schema_path: str, log_dir: str, episodes: int, reward_function_name: str):
    # Create log directory with reward function name
    reward_log_dir = os.path.join(log_dir, reward_function_name)
    os.makedirs(reward_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=reward_log_dir)
    print(f"Logging to TensorBoard at: {reward_log_dir}")

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
    else:
        from citylearn.reward_function import V2GPenaltyReward
        reward_function = V2GPenaltyReward
        print(f"Unknown reward function: {reward_function_name}, using V2GPenaltyReward instead")

    env = CityLearnEnv(schema=schema_path, central_agent=False, reward_function=reward_function)

    model = RLAgent(
        env,
        critic_units=[512, 256, 128],
        actor_units=[256, 128, 64],
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.9773507798877807,
        sigma=0.2264587893937525,
        lr_dual=1e-3,
        steps_between_training_updates=20,
        target_update_interval=100
    )

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
    # Add reward function argument
    parser.add_argument(
        "--reward_function", 
        type=str, 
        default="V2GPenaltyReward", 
        choices=["V2GPenaltyReward", "MARL", "IndependentSACReward", "SolarPenaltyReward"],
        help="Reward function to use"
    )

    args = parser.parse_args()
    main(args.schema_path, args.log_dir, args.episodes, args.reward_function) 