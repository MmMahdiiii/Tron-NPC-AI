# -*- coding: utf-8 -*-

# python imports
import random
from torch import nn
import copy

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


def game_result(new_world, action):
    pass


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)

    def initialize(self):
        pass

    def get_actions(self, world, player):
        actions = []
        for direction in EDirection:
            actions.append(ChangeDirection(direction))
        return actions

    def decide(self):

        # bounded depth MinMax Tree Search
        # calculating heuristics by using the our MagicalBrain
        # choosing the best move

        self.send_command()
        if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
            self.send_command(ActivateWallBreaker())

    # min max tree func
    def min_max_tree(self, depth, world):

        # as while we call this function, always it is my turn to play
        # so we should call max_val_func

        score, move = self.max_val(depth, world)
        return move

    def min_val(self, depth, world):
        # if depth == 0:
        #     return self.heuristic(world)
        # v = float('inf')
        # deep copy world
        # min finding loop operation
        # modified world
        # return v,
        

    def max_val(self, depth, world):
        if depth == 0:
            return self.heuristic(world)
        v = float('-inf')
        depth -= 1
        for action in self.get_actions(world, player = self.my_side):
            # deep copy from world
            new_world = copy.deepcopy(world)
            new_world = game_result(new_world, action, player = self.my_side)
            v2, a2 = self.min_val(depth, new_world)
            if v2 > v:
                v = v2
                move = a2
        return v, move
