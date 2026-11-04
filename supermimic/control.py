"""Action contract shared by real collection, imagination and foot evaluation."""
import math

ACTION_LIMIT = 20.0


def bound_action(action):
    return action.clamp(-ACTION_LIMIT, ACTION_LIMIT)


def return_upper_bound(max_step_reward, gamma):
    """If every reward <= r_max and survival <= 1, discounted return <= r_max/(1-gamma)."""
    if not math.isfinite(max_step_reward) or max_step_reward<=0 or not 0<=gamma<1:
        raise ValueError('A finite positive reward bound and gamma in [0,1) are required')
    return max_step_reward/(1-gamma)


def bound_bootstrap(value, upper_bound):
    # Preserve the legacy Motion objective; only foot tasks supply a proven bound.
    return value if upper_bound is None else value.clamp_max(upper_bound)
