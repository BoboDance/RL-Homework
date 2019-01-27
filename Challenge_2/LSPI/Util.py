from Challenge_2.Common.Util import normalize_state
from Challenge_2.LSPI import Policy


def get_policy_fun(env, policy: Policy, discrete_actions, normalize, low, high) -> callable:
    """
    Return a function handle of the learnt policy
    :param env: Handle of the gym environment
    :param policy: Handle of the policy object
    :param discrete_actions: Numpy array defining the value for each action bin
    :param normalize: Boolean indicating if the environment state space shall be normalized
    :param low: Numpy array describing the lowest numerical values for the env observation
    :param high: Numpy array describing the highest numerical values for the env observation
    :return:
    """
    def policy_fun(obs):
        if normalize:
            obs = normalize_state(env, obs, low, high)

        action_idx = policy.choose_action(obs)
        return discrete_actions[action_idx]

    return policy_fun
