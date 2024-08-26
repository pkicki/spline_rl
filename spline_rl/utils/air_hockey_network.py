import torch

from spline_rl.utils.basic_network import BasicConfigurationNetwork, BasicConfigurationTimeNetwork, BasicLogSigmaNetwork, BasicNetwork
from spline_rl.utils.utils import unpack_data_airhockey

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

class AirHockeyNetwork(BasicNetwork):
    def normalize_input(self, x):
        return normalize_input(self, x)
        
class AirHockeyConfigurationTimeNetwork(BasicConfigurationTimeNetwork):
    def normalize_input(self, x):
        return normalize_input(self, x)

class AirHockeyConfigurationTimeNetworkWrapper(AirHockeyConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(AirHockeyConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])

class AirHockeyLogSigmaNetwork(BasicLogSigmaNetwork):
    def normalize_input(self, x):
        return normalize_input(self, x)

class AirHockeyLogSigmaNetworkWrapper(AirHockeyLogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(AirHockeyLogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])
        

class AirHockeyConfigurationNetwork(BasicConfigurationNetwork):
    def normalize_input(self, x):
        return normalize_input(self, x)

class AirHockeyConfigurationNetworkWrapper(AirHockeyConfigurationNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(AirHockeyConfigurationNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])