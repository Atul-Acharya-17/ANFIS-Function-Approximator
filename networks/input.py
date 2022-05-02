import torch

class InputLayer(torch.nn.Module):

    def __init__(self):
        super(InputLayer, self).__init__()


    def forward(self, x):
        x = torch.tensor(x)
        #x = torch.unsqueeze(x, 0)
        return x