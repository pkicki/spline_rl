import torch

class BasicNetwork(torch.nn.Module):
    def __init__(self, input_space):
        super(BasicNetwork, self).__init__()
        self.input_space = input_space

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        low_high_defined = torch.logical_and(torch.isfinite(low), torch.isfinite(high))
        normalized = torch.where(low_high_defined, 2 * ((x - low) / (high - low + 1e-8)) - 1, x)
        #normalized = (x - low) / (high - low + 1e-8)
        #normalized = 2 * normalized - 1
        return normalized

    def prepare_data(self, x):
        x = self.normalize_input(x)
        return x

    def __call__(self, x):
        x = self.prepare_data(x)
        x = self.fc(x)
        return x


class BasicConfigurationTimeNetwork(BasicNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(BasicConfigurationTimeNetwork, self).__init__(input_space)

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

class BasicConfigurationTimeNetworkWrapper(BasicConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BasicConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])


class BasicLogSigmaNetwork(BasicNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(BasicLogSigmaNetwork, self).__init__(input_space)

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

class BasicLogSigmaNetworkWrapper(BasicLogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BasicLogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])


class BasicFullSigmaNetwork(BasicNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(BasicFullSigmaNetwork, self).__init__(input_space)

        self._init_sigma = init_sigma
        self.N = output_shape[0]

        activation = torch.nn.Tanh()
        W = 128
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, self.N * (self.N + 1) // 2),
        )

    def __call__(self, x):
        x = self.prepare_data(x)
        x = self.fc(x) + torch.log(self._init_sigma)[None]
        return x

class BasicFullSigmaNetworkWrapper(BasicFullSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BasicFullSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])

class BasicConfigurationNetwork(BasicNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(BasicConfigurationNetwork, self).__init__(input_space)

        activation = torch.nn.Tanh()
        W = 256
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0])#, activation,
        )

class BasicConfigurationNetworkWrapper(BasicConfigurationNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(BasicConfigurationNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])