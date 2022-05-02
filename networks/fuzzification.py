import torch

class FuzzificationLayer(torch.nn.Module):

    def __init__(self, a=1, b=2, c=3):
        super(FuzzificationLayer, self).__init__()
        self.register_parameter('a', torch.nn.Parameter(torch.tensor(a, dtype=torch.float)))
        self.register_parameter('b', torch.nn.Parameter(torch.tensor(b, dtype=torch.float)))
        self.register_parameter('c', torch.nn.Parameter(torch.tensor(c, dtype=torch.float)))
        self.b.register_hook(FuzzificationLayer.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        return torch.reciprocal(1 + torch.pow(torch.pow((x - self.c)/self.a, 2), self.b))

