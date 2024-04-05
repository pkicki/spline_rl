import torch

from mushroom_rl.algorithms.policy_search.black_box_optimization.context_builder import ContextBuilder

class IdentityContextBuilder(ContextBuilder):
    def __call__(self, initial_state, **episode_info):
        return torch.tensor(initial_state, dtype=torch.float32)
    