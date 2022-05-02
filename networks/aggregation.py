from matplotlib.pyplot import axis
import torch
from typing import List


class AggregationLayer(torch.nn.Module):

    def __init__(self):
        super(AggregationLayer, self).__init__()

    def forward(self, x:List[torch.Tensor]):

        output = torch.zeros_like(x[0])
        for out in x:
            output += out 
        return output