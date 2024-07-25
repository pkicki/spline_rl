import torch


class BoxPushingNetwork(torch.nn.Module):
    def __init__(self, input_space):
        super(BoxPushingNetwork, self).__init__()
        self.input_space = input_space

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        normalized = (x - low) / (high - low + 1e-8)
        normalized = 2 * normalized - 1
        return normalized

    def prepare_data(self, x):
        x = self.normalize_input(x)
        return x

    
        

class BoxPushingConfigurationTimeNetwork(BoxPushingNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(BoxPushingConfigurationTimeNetwork, self).__init__(input_space)

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
        x = self.prepare_data(x)

        x = self.fc(x)
        q_prototype = self.q_est(x)
        ds_dt_prototype = self.t_est(x)
        return torch.cat([q_prototype, ds_dt_prototype], dim=-1)

class BoxPushingConfigurationTimeNetworkWrapper(BoxPushingConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BoxPushingConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])


class BoxPushingLogSigmaNetwork(BoxPushingNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(BoxPushingLogSigmaNetwork, self).__init__(input_space)

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
        x = self.prepare_data(x)
        x = self.fc(x) + torch.log(self._init_sigma)[None]
        return x

class BoxPushingLogSigmaNetworkWrapper(BoxPushingLogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BoxPushingLogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])