import os
import argparse
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.EVs.maddpgOptimized import MADDPGOptimizedRBC as RLAgent
#from citylearn.agents.EVs.maddpg import MADDPGOptimizedRBC as RLAgent

from citylearn.reward_function import V2GPenaltyReward
#from citylearn.reward_function import SolarPenaltyReward
from citylearn.reward_function import IndependentSACReward
#python Train_CMaddpg.py
#tensorboard --logdir=./logs_Tensorboard

#  lr_dual ≥ lr_critic ≥ lr_actor
def main(schema_path: str, log_dir: str, episodes: int):
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to TensorBoard at: {log_dir}")

    #reward_function = V2GPenaltyReward
    #reward_function = IndependentSACReward
    reward_function = SolarPenaltyReward
    env = CityLearnEnv(schema=schema_path, central_agent=False, reward_function=reward_function)

    model = RLAgent(
        env,
        critic_units=[512, 256, 128],
        actor_units=[256, 128, 64],
        #lr_actor=0.0006343946342268605, #exp1
        #lr_actor= 1e-2, #exp2
        lr_actor= 1e-4,
        #lr_critic=0.0009067117952187151, #exp1
        #lr_critic= 1e-1, exp2
        lr_critic= 1e-4,
        gamma=0.9773507798877807,
        sigma=0.2264587893937525,
        #lr_dual=1e-2, #exp1
        #lr_dual=1e-1, #exp2
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
    print("Training complete. Use `tensorboard --logdir <log_dir>` to visualize results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema_path", type=str,
        default="/home/amalnamm/work/CMDPs/Constrained-EVlearn/data-test/Tiago_dataset/Tiagoschema.json",
        help="Path to the schema.json"
    )
    parser.add_argument("--log_dir", type=str, default="./logs_Tensorboard", help="TensorBoard log directory")
    parser.add_argument("--episodes", type=int, default= 40, help="Number of training episodes")

    args = parser.parse_args()
    main(args.schema_path, args.log_dir, args.episodes)
