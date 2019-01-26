from Challenge_2.Common.Util import normalize_state
from Challenge_2.LSPI import Policy


def get_policy_fun(env, policy: Policy, normalize, discrete_actions):
    def policy_fun(obs):
        if normalize:
            obs = normalize_state(env, obs)

        action_idx = policy.choose_action(obs)
        return discrete_actions[action_idx]

    return policy_fun