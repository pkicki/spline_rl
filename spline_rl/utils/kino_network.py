import torch

from spline_rl.utils.utils import unpack_data_kinodynamic


class KinoNetwork(torch.nn.Module):
    def __init__(self, input_space):
        super(KinoNetwork, self).__init__()
        self.input_space = input_space

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        normalized = (x - low) / (high - low)
        normalized = 2 * normalized - 1
        return normalized

    def prepare_data(self, x):
        q0, qd, dq0, dqd, ddq0, ddqd = unpack_data_kinodynamic(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    
        

class KinoConfigurationTimeNetwork(KinoNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(KinoConfigurationTimeNetwork, self).__init__(input_space)

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

class KinoConfigurationTimeNetworkWrapper(KinoConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(KinoConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])


class KinoLogSigmaNetwork(KinoNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(KinoLogSigmaNetwork, self).__init__(input_space)

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
        #x = torch.log(self._init_sigma)[None].to(torch.float64)
        x = self.fc(x) + torch.log(self._init_sigma)[None]
        return x

class KinoLogSigmaNetworkWrapper(KinoLogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(KinoLogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])