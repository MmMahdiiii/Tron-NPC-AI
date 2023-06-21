# -*- coding: utf-8 -*-

# python imports
import random
from torch import nn

# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker

# pytorch imports
import torch
from torch import nn


class Soul(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(Soul, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.layers = []
        prev_layer_size = input_size
        for i, layer_size in enumerate(hidden_layers):
            layer = nn.Linear(prev_layer_size, layer_size)
            self.add_module(f"hidden_layer_{i}", layer)
            self.layers.append(layer)
            prev_layer_size = layer_size

        self.output_layer = nn.Linear(prev_layer_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)

    def initialize(self):
        pass

    def decide(self):
        # bounded depth MinMax Tree Search
        # calculating heuristics by using the our MagicalBrain
        # choosing the best move

        self.send_command()
        if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
            self.send_command(ActivateWallBreaker())
