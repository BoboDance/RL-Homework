import numpy as np


def create_initial_samples(env, memory, count, discrete_actions):
    last_observation = env.reset()
    samples = 0
    while samples < count:
        # choose a random action
        action_idx = np.random.choice(range(len(discrete_actions)), 1)
        action = discrete_actions[action_idx]
        # do the step
        observation, reward, done, _ = env.step(action)

        memory.push((*last_observation, *action_idx, reward, *observation, done))

        samples += 1
        last_observation = observation

        if done:
            last_observation = env.reset()