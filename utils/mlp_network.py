import torch.nn.functional as F
from torch import nn

# An argument we read from pytorch documentation
# We choose input_mlp_size to be 12544 because we read on the model
# https://pytorch.org/vision/0.8/_modules/torchvision/models/detection/faster_rcnn.html


class MLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, input_mpl_size=12544, representation_size=1024, dropout=0.25):
        super().__init__()

        self.fc6 = nn.Linear(input_mpl_size, representation_size)
        self.dropout = nn.Dropout(dropout)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))

        return x
