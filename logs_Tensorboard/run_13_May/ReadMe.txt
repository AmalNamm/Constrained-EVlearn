In this run we have a smooth inequality cost function; without other changes
No cost/reward scaling, or Teglu changes. 
Hyperparameters: 
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