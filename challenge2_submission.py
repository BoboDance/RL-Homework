"""
Submission template for Programming Challenge 2: Approximate Value Based Methods.


Fill in submission info and implement 4 functions:

- load_dqn_policy
- train_dqn_policy
- load_lspi_policy
- train_lspi_policy

Keep Monitor files generated by Gym while learning within your submission.
Example project structure:

challenge2_submission/
  - challenge2_submission.py
  - dqn.py
  - lspi.py
  - dqn_eval/
  - dqn_train/
  - lspi_eval/
  - lspi_train/
  - supplementary/

Directories `dqn_eval/`, `dqn_train/`, etc. are autogenerated by Gym (see below).
Put all additional results into the `supplementary` directory.

Performance of the policies returned by `load_xxx_policy` functions
will be evaluated and used to determine the winner of the challenge.
Learning progress and learning algorithms will be checked to confirm
correctness and fairness of implementation. Supplementary material
will be manually analyzed to identify outstanding submissions.
"""
import pickle

import gym
import numpy as np
import torch

from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNSwingShortModel import DQNSwingShortModel
from Challenge_2.LSPI.BasisFunctions.FourierBasis import FourierBasis
from Challenge_2.LSPI.LSPI import LSPI
from Challenge_2.LSPI.Policy import Policy
import torch.nn as nn

info = dict(
    group_number=16,  # change if you are an existing seminar/project group
    authors="Fabian Otto; Johannes Czech; Jannis Weil",
    description="Explain what your code does and how. "
                "Keep this description short "
                "as it is not meant to be a replacement for docstrings "
                "but rather a quick summary to help the grader.")


def load_dqn_policy():
    """
    Load pretrained DQN policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleSwingShort-v0` env.

    :return: function pi: s -> a
    """

    # this is required in order to determine input and output shapes
    env = gym.make("CartpoleStabShort-v0")

    # discrete actions
    min_action = -5
    max_action = 5
    nb_bins = 7
    discrete_actions = np.linspace(min_action, max_action, nb_bins)

    Q = DQNSwingShortModel(env, discrete_actions, optimizer=None, lr=0)
    Q.load_state_dict(torch.load("./checkpoints/Policies/best_weights.pth"))

    from Challenge_2.DQN.Util import get_policy_fun
    return get_policy_fun(env, Q, normalize=False)


def train_dqn_policy(env):
    """
    Execute your implementation of the DQN learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartpoleSwingShort-v0")
    env.seed(seed)

    # hyperparameter selection

    # discrete actions
    min_action = -5
    max_action = 5
    nb_bins = 7
    discrete_actions = np.linspace(min_action, max_action, nb_bins)

    # epsilon greed parameters for exp decay
    eps_start = 1
    eps_end = 0.0
    eps_decay = 5e4

    # usual basic hyperparameters
    lr = 1e-4
    memory_size = int(1e6)
    gamma = 0.999
    max_episodes = 100
    max_episode_length = 1e4
    minibatch_size = 1024
    target_model_update_steps = 5000
    optimizer = "rmsprop"
    # loss = nn.SmoothL1Loss()
    loss = nn.MSELoss()

    normalize = False

    lr_scheduler = None
    # lr_scheduler = StepLR(Q.optimizer, max_episode_length, 0.5)  # None
    # lr_scheduler = CosineAnnealingLR(Q.optimizer, 10) #T_max=max_episode_length)

    anti_sucide = False
    edge_fear_threshold = .3
    use_tensorboard = False

    Q = DQNSwingShortModel(env, discrete_actions, optimizer=optimizer, lr=lr)

    dqn = DQN(env, Q, memory_size=memory_size, initial_memory_count=minibatch_size, minibatch_size=minibatch_size,
              target_model_update_steps=target_model_update_steps, gamma=gamma,
              eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, max_episodes=max_episodes,
              max_steps_per_episode=max_episode_length, lr_scheduler=lr_scheduler, loss=loss, normalize=normalize,
              anti_suicide=anti_sucide, edge_fear_threshold=edge_fear_threshold, use_tensorboard=use_tensorboard)

    dqn.train()

    # load best model from this training run
    # this is a different path from above, no test in here, this ensures, we do not just have the last Q
    Q.load_state_dict(torch.load("./checkpoints/best_weights.pth"))

    from Challenge_2.DQN.Util import get_policy_fun
    return get_policy_fun(env, Q, normalize=False)


def load_lspi_policy():
    """
    Load pretrained LSPI policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleStabShort-v0` env.

    :return: function pi: s -> a
    """

    # this is required in order to normalize the inputs
    env = gym.make("CartpoleStabShort-v0")

    # Our discrete actions
    discrete_actions = np.linspace(-5, 5, 3)

    # Normalization of the observation space
    low = np.array(list(env.observation_space.low[:3]) + [-4, -20])
    high = np.array(list(env.observation_space.high[:3]) + [4, 20])

    policy = pickle.load(open("./Challenge_2/LSPI/Policies/CartpoleStabShort-v0-20k.pkl", "rb"))

    from Challenge_2.LSPI.Util import get_policy_fun
    return get_policy_fun(env, policy, discrete_actions, True, low, high)


def train_lspi_policy(env):
    """
    Execute your implementation of the LSPI learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """

    # Set the seed to get the same features and training samples (our features are generated randomly)
    # Note: As the seed influences the features, more features/samples may be needed for different seeds
    seed = 2
    np.random.seed(seed)
    # env.seed(seed)
    dim_obs = env.observation_space.shape[0]

    # Our discrete actions
    discrete_actions = np.linspace(-5, 5, 3)

    # Normalization of the observation space
    low = np.array(list(env.observation_space.low[:3]) + [-4, -20])
    high = np.array(list(env.observation_space.high[:3]) + [4, 20])

    # Create a policy using the fourier base function
    basis_function = FourierBasis(input_dim=dim_obs, n_features=100, n_actions=len(discrete_actions))
    policy = Policy(basis_function=basis_function, n_actions=len(discrete_actions), eps=0)

    # Start training using LSPI
    lspi = LSPI(env, policy, discrete_actions, True, low, high, 0.99, 1e-5, 25000, full_episode=True)
    lspi.train(policy_step_episodes=1, do_render=False)

    from Challenge_2.LSPI.Util import get_policy_fun

    return get_policy_fun(env, policy, discrete_actions, True, low, high)


# ==== Example evaluation
def main():
    import gym
    from gym.wrappers.monitor import Monitor
    import quanser_robots

    def evaluate(env, policy, num_evlas=25):
        ep_returns = []
        for eval_num in range(num_evlas):
            episode_return = 0
            dones = False
            obs = env.reset()
            while not dones:
                action = policy(obs)
                obs, rewards, dones, info = env.step(action)
                episode_return += rewards
            ep_returns.append(episode_return)
        return ep_returns

    def render(env, policy):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            act = policy(obs)
            obs, _, done, _ = env.step(act)

    def check(env, policy):
        render(env, policy)
        ret_all = evaluate(env, policy)
        print(np.mean(ret_all), np.std(ret_all))
        env.close()

    # DQN I: Check learned policy
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_eval')
    policy = load_dqn_policy()
    check(env, policy)

    # DQN II: Check learning procedure
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_train', video_callable=False)
    policy = train_dqn_policy(env)
    check(env, policy)

    # LSPI I: Check learned policy
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_eval')
    policy = load_lspi_policy()
    check(env, policy)

    # LSPI II: Check learning procedure
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_train', video_callable=False)
    policy = train_lspi_policy(env)
    check(env, policy)


if __name__ == '__main__':
    main()
