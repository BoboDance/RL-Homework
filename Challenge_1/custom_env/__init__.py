from gym.envs.registration import register

register(
    id='PendulumCustom-v0',
    entry_point='Challenge_1.custom_env.PendulumEnvCustom:PendulumEnvCustom',
    max_episode_steps=200,
)
