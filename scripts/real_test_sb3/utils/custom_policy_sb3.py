
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
from torch.nn.modules.linear import Linear

import torchvision.models as pre_models
import numpy as np
import torch.nn.functional as F

'''
Here we provide 5 feature extractor networks

1. No_CNN
    No CNN layers
    Only maxpooling layer to generate 25 features

2. CNN_GAP
    3 layers of CNN
    finished by AvgPool2d
    1*8 -> 8*16 -> 16*25

3. CNN_GAP_BN
    3 layers of CNN with BN for each CNN layer
    finished by AvgPool2d

4. CNN_FC
    3 layers of CNN
    finished by Flatten
    FC is used to get CNN features (960 100 25)

5. CNN_MobileNet
    Using a pre-trained MobileNet as feature generator
    finished by Flatten (576 -> 25)
'''


class No_CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(No_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5
        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(16, 20)),
            # nn.MaxPool2d(kernel_size=(26, 33)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]
        # print(cnn_feature)
        # print(self.feature_num_state)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        # print(x)
        self.feature_all = x  # use  to update feature before FC

        return x


class CNN_GAP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 48]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

        # nn.init.kaiming_normal_(self.conv1[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv2[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv3[0].weight, a=0, mode='fan_in')
        # nn.init.constant(self.conv1[0].bias, 0.0)
        # nn.init.constant(self.conv2[0].bias, 0.0)
        # nn.init.constant(self.conv3[0].bias, 0.0)

        # nn.init.xavier_uniform(self.conv1[0].weight)
        # nn.init.xavier_uniform(self.conv2[0].weight)
        # nn.init.xavier_uniform(self.conv3[0].weight)
        # self.conv1[0].bias.data.fill_(0)
        # self.conv2[0].bias.data.fill_(0)
        # self.conv3[0].bias.data.fill_(0)
        # self.soft_max_layer = nn.Softmax(dim=1)
        # self.batch_norm_layer = nn.BatchNorm1d(16, affine=False)
        
        # self.linear = self.cnn

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3) # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2) # [1, 8]
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        
        return x


class CNN_GAP_BN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_BN, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 48]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3) # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2) # [1, 8]
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        
        return x


class CustomNoCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(CustomNoCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5
        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(16, 20)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        # print(x)
        self.feature_all = x  # use  to update feature before FC
        
        return x


class CNN_FC(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_FC, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # Input image: 80*100
        # Output: 16 CNN features + n state features 
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 40, 48]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]

            # nn.BatchNorm2d(8),
            nn.Flatten(),   # 960
            # nn.AvgPool2d(kernel_size=(10, 12), stride=1)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None][:, 0:1, :, :]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 100),
            nn.ReLU(),
            nn.Linear(100, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.linear(self.cnn(depth_img))
        # cnn_feature = cnn_feature.squeeze(dim=3) # [1, 8, 1]
        # cnn_feature = cnn_feature.squeeze(dim=2) # [1, 8]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        # print(x)
        
        return x


class CNN_MobileNet(BaseFeaturesExtractor):
    '''
    Using part of mobile_net_v3_small to generate features from depth image
    '''
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_MobileNet, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.mobilenet_v3_small = pre_models.mobilenet_v3_small(pretrained=True)

        self.part = self.mobilenet_v3_small.features

        # freeze part parameters
        for param in self.part.parameters():
            param.requires_grad = False
        
        self.gap_layer = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Sequential(
            nn.Linear(576, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.25)
        )
        self.linear_small = nn.Sequential(
            nn.Linear(576, self.feature_num_cnn),
            nn.Tanh(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.25)
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        # change input image to (None, 3, 100, 80)
        depth_img_stack = depth_img.repeat(1, 3, 1, 1)  # notion: this repeat is used for tensor  # (1, 3, 80 ,100)

        self.last_cnn_output = self.part(depth_img_stack)        # [1, 576, 3, 4]
        self.gap_layer_out = cnn_feature = self.gap_layer(self.last_cnn_output) # [1, 576, 1, 1]

        cnn_feature = cnn_feature.squeeze(dim=3) # [1, 576, 1]
        cnn_feature = cnn_feature.squeeze(dim=2) # [1, 576]
        cnn_feature = self.linear_small(cnn_feature)  # [1, 32]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state] # [1, 2]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        # print(x)
        
        return x


class CNN_GAP_new(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_new, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # input size (100, 80)
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap_layer = nn.AvgPool2d(kernel_size=(8, 10), stride=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.pool(F.relu(self.conv1(depth_img)))         # 1, 8, 38, 48
        self.layer_2_out = self.pool(F.relu(self.conv2(self.layer_1_out)))  # 1, 8, 18, 23
        self.layer_3_out = self.pool(F.relu(self.conv3(self.layer_2_out)))  # 1, 16, 8, 10
        self.gap_layer_out = self.gap_layer(self.layer_3_out)               # 1, 16, 1, 1

        cnn_feature = self.gap_layer_out  # [1, 16, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 16, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 16]
        # cnn_feature = th.clamp(cnn_feature, -1, 2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        
        return x