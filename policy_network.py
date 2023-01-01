import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class CNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        num_class: int = 100,
    ):
        """
        A shared Encoder of both the Policy and Value networks.
        This Features Extractor will create an embedding vector from the inputs, and use that vector as the input
            of the policy and value networks.

        :param observation_space: gym.spaces.Spaces
        :param features_dim: final shape of the embedding vecotr
        :param num_class: number of classes in the dataset
        """
        super(CNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["saliency"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels, 150, kernel_size=(3, 3), stride=(1, 1), padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(150, 100, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        self.logits_layer = nn.Sequential(
            nn.Linear(num_class, features_dim), nn.Flatten()
        )
        with torch.no_grad():
            embed_dim = self.cnn(
                torch.as_tensor(observation_space["saliency"].sample()[None]).float()
            ).shape[-1]
            embed_dim += self.logits_layer(
                torch.as_tensor(observation_space["logits"].sample()[None]).float()
            ).shape[-1]

        self.linear = nn.Sequential(nn.Linear(embed_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        saliency = self.cnn(observations["saliency"])
        logits = self.logits_layer(observations["logits"])
        cat = torch.cat((saliency, logits), dim=1)
        return self.linear(cat)
    

class SaliencyGuidedRLCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Dict, 
                 features_dim: int = 256,
                 **kwargs):
        super(SaliencyGuidedRLCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space["original_saliency"].shape[0]
        
        self.original_cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels, 150, kernel_size=(3, 3), stride=(1, 1), padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(150, 100, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        
        self.perm_cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels, 150, kernel_size=(3, 3), stride=(1, 1), padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(150, 100, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        
        with torch.no_grad():
            embed_dim = self.original_cnn(
                torch.as_tensor(observation_space["original_saliency"].sample()[None]).float()
            ).shape[-1]
            
            embed_dim += self.perm_cnn(
                torch.as_tensor(observation_space["perm_saliency"].sample()[None]).float()
            ).shape[-1]
            
        self.linear = nn.Sequential(nn.Linear(embed_dim, features_dim), nn.ReLU())
        
    def forward(self, observations):
        original_saliency = self.original_cnn(observations["original_saliency"])
        perm_saliency = self.perm_cnn(observations["perm_saliency"])
        features = torch.cat([original_saliency, perm_saliency], dim=1)
        return self.linear(features)


if __name__ == "__main__":
    patch_size = 4
    c = SaliencyGuidedRLCNNExtractor(gym.spaces.Dict(dict(
            original_saliency=gym.spaces.Box(0.0, 1.0, shape=(100, patch_size, patch_size)),
            perm_saliency=gym.spaces.Box(0.0, 1.0, shape=(100, patch_size, patch_size)),
        )))
    print(c(dict(original_saliency=torch.rand(1, 100, patch_size, patch_size),
                 perm_saliency=torch.rand(1, 100, patch_size, patch_size))))