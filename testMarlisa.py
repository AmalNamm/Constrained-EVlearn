from citylearn.citylearn import CityLearnEnv
from citylearn.agents.marlisa import MARLISA as RLAgent

dataset_name = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(dataset_name)
print(env.get_building_information())

agents = RLAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    building_information=env.get_building_information(),
    observation_names=env.observation_names,
)
print("AHAH")
episodes = 1 # number of training episodes

# train agents
for e in range(episodes):
    observations = env.reset()

    while not env.done:
        actions = agents.select_actions(observations)

        # apply actions to env
        next_observations, rewards, _, _ = env.step(actions)

        # update policies
        agents.add_to_buffer(observations, actions, rewards, next_observations, done=env.done)
        observations = [o for o in next_observations]

    # print cost functions at the end of episode
    print(f'Episode: {e}')

    for n, nd in env.evaluate().groupby('name'):
        nd = nd.pivot(index='name', columns='cost_function', values='value').round(3)
        print(n, ':', nd.to_dict('records'))