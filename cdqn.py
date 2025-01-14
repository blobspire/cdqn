import torch
from torch import nn
import torch.nn.functional as F

class CDQN(nn.Module):


    def __init__(self, input_shape, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(CDQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        conv_out_size = 64 * 7 * 7 # TODO parameterize
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256) # TODO what should i set number of nodes to instead of hardcoding
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256) # TODO likewise
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten layer
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(x)

        return Q


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = CDQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)