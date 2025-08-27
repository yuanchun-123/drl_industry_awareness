"""
valuation.py

A PyTorch MLP network that maps state/observation to a scalar value estimate.
"""
import torch
import torch.nn as nn

class ValuationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[256,128,64], activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)