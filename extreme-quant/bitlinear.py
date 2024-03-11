import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantization function
def quantization_function(W, act=False):
    if act:
        Wf = torch.clamp(W, 0, 2**8 - 1)  # Clamp activations to 8-bit range
    else:
        gamma = torch.mean(torch.abs(W), dim=(0, 1), keepdim=True)  # Calculate average absolute value
        W_scaled = W / gamma  # Scale weight matrix
        W_scaled = W_scaled + torch.rand_like(W_scaled) * 1e-8  # Add small noise to break ties
        Wf = torch.round(torch.clamp(W_scaled, -1, 1))  # Round and clip to {-1, 0, +1}
    return Wf


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        # self.weight.data = quantization_function(self.weight, act=False).to(input.device)
        out = nn.functional.linear(input, self.weight)
        if self.bias is not None:
            out += self.bias.view(1, -1)
        return out


class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.fc1 = BitLinear(784, 256)
        self.fc2 = BitLinear(256, 128)
        self.fc3 = BitLinear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x