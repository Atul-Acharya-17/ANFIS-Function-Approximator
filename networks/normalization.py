from matplotlib.pyplot import axis
import torch
from typing import List


class NormalizationLayer(torch.nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x:List[torch.Tensor]):
        x = torch.stack(x)
        sum = torch.sum(x, axis=0)
        x = x/sum
        x = torch.permute(x, (1, 0))
        return x