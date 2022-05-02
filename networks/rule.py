import torch
from typing import List


class RuleLayer(torch.nn.Module):

    def __init__(self):
        super(RuleLayer, self).__init__()

    def forward(self, x:List[torch.Tensor]):
        product = torch.ones_like(x[0])

        for out in x:
            product = product * out

        return product
