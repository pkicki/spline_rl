import torch
from utils.utils import unpack_data_airhockey


class AirHockeyNetwork(torch.nn.Module):
    def __init__(self, input_space):
        super(AirHockeyNetwork, self).__init__()
        self.input_space = input_space

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        normalized = (x - low) / (high - low)
        normalized = 2 * normalized - 1
        # move puck position taking into account its velocity
        #x[:, 0] += x[:, 3] * 0.55
        #x[:, 1] += x[:, 4] * 0.55
        #x[:, 3:6] = 0. # to simulate no information about the puck velocity
        normalized[:, 0] = (x[:, 0] - 1.51) / (1.948 / 2. - 0.03165)
        normalized[:, 1] = x[:, 1] / (1.038 / 2. - 0.03165)
        return normalized

    def prepare_data(self, x):
        puck, puck_dot, q0, qd, dq0, dqd, ddq0, ddqd, _ = unpack_data_airhockey(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    
        

class ConfigurationTimeNetwork(AirHockeyNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(ConfigurationTimeNetwork, self).__init__(input_space)

        activation = torch.nn.Tanh()
        W = 256
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
        )

        self.q_est = torch.nn.Sequential(
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0])#, activation,
        )

        self.t_est = torch.nn.Sequential(
            torch.nn.Linear(W, output_shape[1]),
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)

        x = self.fc(x)
        q_prototype = self.q_est(x)
        ds_dt_prototype = self.t_est(x)
        return torch.cat([q_prototype, ds_dt_prototype], dim=-1)

class ConfigurationTimeNetworkWrapper(ConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(ConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])


class LogSigmaNetwork(AirHockeyNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(LogSigmaNetwork, self).__init__(input_space)

        self._init_sigma = init_sigma

        activation = torch.nn.Tanh()
        W = 128
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0]),
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        x = torch.log(self._init_sigma)[None].to(torch.float64)
        #x = self.fc(x) + torch.log(self._init_sigma)[None]
        return x

class LogSigmaNetworkWrapper(LogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(LogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])
        

class ConfigurationNetwork(AirHockeyNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(ConfigurationNetwork, self).__init__(input_space)

        activation = torch.nn.Tanh()
        W = 256
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0])#, activation,
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        x = self.fc(x)
        return x

class ConfigurationNetworkWrapper(ConfigurationNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(ConfigurationNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])