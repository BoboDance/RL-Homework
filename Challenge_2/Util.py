def create_initial_samples(env, memory, count):
    last_observation = env.reset()
    samples = 0
    action = env.action_space.sample()
    while samples < count:
        action = env.action_space.sample()  # np.clip(np.random.normal(action, 1), env.action_space.low, env.action_space.high)
        observation, reward, done, info = env.step(action)
        memory.push((*last_observation, *action, reward, *observation, done))
        samples += 1

        if done:
            last_observation = env.reset()
            action = env.action_space.sample()


def get_y(transitions, discount_factor, old_model):
    y = np.empty((len(transitions), 1))

    for i, replay_entry in enumerate(transitions):
        if replay_entry[DONE_INDEX]:
            # if the episode is done after this step, simply use the reward
            y[i] = replay_entry[REWARD_INDEX]
        else:
            # otherwise look ahead one step
            next_obs = replay_entry[NEXT_OBS_INDEX:NEXT_OBS_INDEX + dim_obs]
            y[i] = replay_entry[REWARD_INDEX] + discount_factor * old_model.get_best_value(next_obs, discrete_actions)

    return y