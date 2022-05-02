from matplotlib.pyplot import axis
import torch
from typing import List


class DefuzzificationLayer(torch.nn.Module):

    def __init__(self, input_features):
        super(DefuzzificationLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_features, out_features=1, bias=True)

    def forward(self, x:List[torch.Tensor], weight):
        x = torch.stack(x)
        x = torch.permute(x, (1,0))
        f = self.linear(x.type(torch.FloatTensor))
        out = f*weight
        return out