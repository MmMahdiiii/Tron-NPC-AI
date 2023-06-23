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

    def __init__(self, world):
        super(AI, self).__init__(world)
        self.nn = None

    def initialize(self):
        with open("nn_neat", "rb") as f:
            self.nn = pickle.load(f)
        if self.nn is None:
            raise Exception("nn is None")

    def heuristic(self, world):
        # calculating distances of the agent in 8 directions to the nearest wall
        # if was enemy wall, then make enemy = 1
        x, y = world.agents[self.my_side].position.x, world.agents[self.my_side].position.y
        pointers = [Position(x, y)] * 8
        found_walls = [False] * 8
        distances = [0] * 8
        enemies = [False] * 8
        width = len(world.board[0])
        height = len(world.board)

        def is_enemy_wall(X, Y):
            if self.other_side == 'Blue':
                return world.board[Y][X] == ECell.BlueWall
            else:
                return world.board[Y][X] == ECell.YellowWall

        def stop_condition():
            return all(found_walls)

        def step():
            for i in range(8):
                if found_walls[i]:
                    continue
                if pointers[i].x < 0 or pointers[i].x >= width or pointers[i].y < 0 or pointers[i].y >= height:
                    found_walls[i] = True
                    continue
                if world.board[pointers[i].y][pointers[i].x] != ECell.Empty:
                    found_walls[i] = True
                    if i % 2 == 0:
                        distances[i] = abs(pointers[i].x - pointers[i - 1].x) + abs(
                            pointers[i].y - pointers[i - 1].y)
                    else:
                        distances[i] = math.sqrt((pointers[i].x - pointers[i + 1].x) ** 2 + \
                                                    (pointers[i].y - pointers[i + 1].y) ** 2)
                    if is_enemy_wall(pointers[i].x, pointers[i].y):
                        enemies[i] = True
                    continue
                if i == 0:
                    pointers[i].x += 1
                elif i == 1:
                    pointers[i].x += 1
                    pointers[i].y += 1
                elif i == 2:
                    pointers[i].y += 1
                elif i == 3:
                    pointers[i].x -= 1
                    pointers[i].y += 1
                elif i == 4:
                    pointers[i].x -= 1
                elif i == 5:
                    pointers[i].x -= 1
                    pointers[i].y -= 1
                elif i == 6:
                    pointers[i].y -= 1
                elif i == 7:
                    pointers[i].x += 1
                    pointers[i].y -= 1
        
        while not stop_condition():
            step()

        # concatenating distances, enemies and wall_breaker_cooldown and wall_breaker_cooldown and health
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
        self.send_command(ChangeDirection(random.choice(list(EDirection))))
        if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
            self.send_command(ActivateWallBreaker())
