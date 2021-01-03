import torch
import torch.nn.functional as F
from torch import nn


# A simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, activation=nn.Sigmoid()):
        super().__init__()
        self.activation = activation
        self.layer_1 = nn.Linear(16, 2)

    def forward(self, x):
        predictions = self.layer_1(x)
        return predictions

# A feedforward network with one hidden layer
class FeedForwardNetwork(nn.Module):
    def __init__(self, activation=nn.Sigmoid()):
        super().__init__()
        self.activation = activation
        self.layer_1 = nn.Linear(16, 10)
        self.layer_2 = nn.Linear(10, 2)

    def forward(self, x):
        output_1 = self.activation(self.layer_1(x))
        predictions = self.layer_2(output_1)
        return predictions

# A CNN for 6 x 6 x 3 images
class MatchNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,stride=1)
        self.fc1 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(10, 2)
    def forward(self,x):
        x= self.conv1(x)
        x= F.relu(x)
        x= self.conv2(x)
        x= F.relu(x)
        x= F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output