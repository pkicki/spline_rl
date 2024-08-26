from spline_rl.utils.basic_network import BasicNetwork
import torch

from spline_rl.utils.air_hockey_network import AirHockeyNetwork

class BasicValueNetwork(BasicNetwork):
    def __init__(self, input_space, bias):
        super().__init__(input_space)
        W = 128
        self.bias = bias

        activation = torch.nn.Tanh()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_space.shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, 1),
        )

    def __call__(self, x):
        return super().__call__(x) + self.bias

class AirHockeyValueNetwork(AirHockeyNetwork):
    def __init__(self, input_space, bias):
        super().__init__(input_space)
        W = 128
        self.bias = bias

        activation = torch.nn.Tanh()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_space.shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, 1),
        )

    def __call__(self, x):
        return super().__call__(x) + self.bias