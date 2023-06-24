# -*- coding: utf-8 -*-

# python imports
import random
import math
import pickle

# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker

# neat imports
from neat import nn as neat_nn


class AI(RealtimeAI):

    def __init__(self, world, nn):
        super(AI, self).__init__(world)
        self.nn = nn

    # def initialize(self):
    #     with open("nn_neat", "rb") as f:
    #         self.nn = pickle.load(f)
    #     if self.nn is None:
    #         raise Exception("nn is None")

    def heuristic(self, world):
        # calculating distances of the agent in 8 directions to the nearest wall
        # if was enemy wall, then make enemy = 1

        def is_enemy_wall(X, Y):
            if self.other_side == 'Blue':
                return world.board[X][Y] == ECell.BlueWall
            else:
                return world.board[X][Y] == ECell.YellowWall

        print(world.agents[self.my_side].position.x, world.agents[self.my_side].position.y)

        x, y = world.agents[self.my_side].position.x, world.agents[self.my_side].position.y

        pos = []
        for i in range(8):
            pos.append([y, x])
        distances = [0] * 8
        enemies = [0] * 8

        # down
        pos[0][0] += 1
        while world.board[pos[0][0]][pos[0][1]] == ECell.Empty:
            pos[0][0] += 1

        # left down
        pos[1][0] += 1
        pos[1][1] -= 1
        while world.board[pos[1][0]][pos[1][1]] == ECell.Empty:
            pos[1][0] += 1
            pos[1][1] -= 1

        # left
        pos[2][1] -= 1
        while world.board[pos[2][0]][pos[2][1]] == ECell.Empty:
            pos[2][1] -= 1

        # left up
        pos[3][0] -= 1
        pos[3][1] -= 1
        while world.board[pos[3][0]][pos[3][1]] == ECell.Empty:
            pos[3][0] -= 1
            pos[3][1] -= 1

        # up
        pos[4][0] -= 1
        while world.board[pos[4][0]][pos[4][1]] == ECell.Empty:
            pos[4][0] -= 1

        # right up
        while world.board[pos[5][0]][pos[5][1]] == ECell.Empty:
            pos[5][0] -= 1
            pos[5][1] += 1

        # down
        pos[6][1] += 1
        while world.board[pos[6][0]][pos[6][1]] == ECell.Empty:
            pos[6][1] += 1

        # right down
        pos[7][0] += 1
        pos[7][1] += 1
        while world.board[pos[7][0]][pos[7][1]] == ECell.Empty:
            pos[7][0] += 1
            pos[7][1] += 1

        for i in range(8):
            distances[i] = math.sqrt((pos[i][1] - x) ** 2 + (pos[i][0] - y) ** 2)
            if is_enemy_wall(pos[i][0], pos[i][1]):
                enemies[i] = 1

        information = distances + enemies + [world.agents[self.my_side].wall_breaker_cooldown,
                                             world.agents[self.my_side].wall_breaker_rem_time,
                                             world.agents[self.my_side].health]
        onehot_direction = [0] * 4
        onehot_direction[world.agents[self.my_side].direction.value] = 1
        information += onehot_direction
        return self.nn.activate(information)

    def decide(self):
        # bounded depth MinMax Tree Search
        # calculating heuristics by using the our MagicalBrain
        # choosing the best move
        # TODO
        direction = self.heuristic(self.world)[0:4]
        active_wall_breaker = self.heuristic(self.world)[4]
        # if direction is opposite to current direction, then choose 2nd best move
        best_move = int(direction.index(max(direction)))
        if (best_move + 2) % 4 == self.world.agents[self.my_side].direction.value:
            # find best second move
            direction[best_move] = -float('inf')

        best_move = int(direction.index(max(direction)))
        self.send_command(ChangeDirection(EDirection(best_move)))
        if self.world.agents[self.my_side].wall_breaker_cooldown == 0 and active_wall_breaker > 0.5:
            self.send_command(ActivateWallBreaker())
