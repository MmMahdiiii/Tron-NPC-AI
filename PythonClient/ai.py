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


# todo: create a class for handling the game including game_result


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)

    def calculate_score(self, new_world, is_us, current_agent):
        added_score = 0
        opp_added_score = 0
        agent_side = self.my_side if is_us else self.other_side
        wall_breaker_rem_time = current_agent.wall_breaker_rem_time
        wall_breaker_cooldown = current_agent.wall_breaker_cooldown

        destination = new_world.board[current_agent.position.y][current_agent.position.x]

        if destination == ECell.Empty:
            added_score += new_world.constants.wall_score_coefficient
        else:
            if (is_us and self.my_side == 'Yellow' and destination == ECell.YellowWall) or \
                    (not is_us and self.my_side == 'Yellow' and destination == ECell.BlueWall) or \
                    (is_us and self.my_side == 'Blue' and destination == ECell.BlueWall) or \
                    (not is_us and self.my_side == 'Blue' and destination == ECell.YellowWall):
                if wall_breaker_cooldown == 0 and wall_breaker_rem_time > 0:
                    added_score += 0
                    current_agent.wall_breaker_rem_time -= 1
                else:
                    if current_agent.health > 1:
                        added_score += 0
                        current_agent.health -= 1
                    else:
                        added_score += new_world.constants.my_wall_crash_score
            elif (is_us and self.my_side == 'Yellow' and destination == ECell.BlueWall) or \
                    (not is_us and self.my_side == 'Yellow' and destination == ECell.YellowWall) or \
                    (is_us and self.my_side == 'Blue' and destination == ECell.YellowWall) or \
                    (not is_us and self.my_side == 'Blue' and destination == ECell.BlueWall):
                added_score += new_world.constants.wall_score_coefficient
                opp_added_score -= new_world.constants.wall_score_coefficient
                if wall_breaker_cooldown == 0 and wall_breaker_rem_time > 0:
                    new_world.agents[agent_side].wall_breaker_rem_time -= 1
                else:
                    if current_agent.health > 1:
                        current_agent.health -= 1
                    else:
                        added_score += new_world.constants.enemy_wall_crash_score
            elif destination == ECell.AreaWall:
                added_score += new_world.constants.area_wall_crash_score

        return added_score, opp_added_score, current_agent

    def game_result(self, new_world, action, is_us):
        # creating agent my_side
        current_agent = copy.deepcopy(new_world.agents[self.my_side]) if is_us else \
            copy.deepcopy(new_world.agents[self.other_side])

        # direction change
        if action == "up":
            current_agent.position.y -= 1
            current_agent.direction = EDirection.Up
        elif action == "down":
            current_agent.position.y += 1
            current_agent.direction = EDirection.Down
        elif action == "left":
            current_agent.position.x -= 1
            current_agent.direction = EDirection.Left
        elif action == "right":
            current_agent.position.x += 1
            current_agent.direction = EDirection.Right

        # calculate score
        current_score = new_world.scores[self.my_side] if is_us else new_world.scores[self.other_side]
        added_score_curr_agent, added_score_opp_curr_agent, current_agent = self.calculate_score(new_world, is_us,
                                                                                            current_agent)
        final_score = current_score + added_score_curr_agent
        if is_us:
            new_world.scores[self.my_side] = final_score
            new_world.scores[self.other_side] += added_score_opp_curr_agent
        else:
            new_world.scores[self.other_side] = final_score
            new_world.scores[self.my_side] += added_score_opp_curr_agent

        # update wall breaker
        if current_agent.wall_breaker_rem_time == 0:
            current_agent.wall_breaker_cooldown = new_world.constants.wall_breaker_cooldown
            current_agent.wall_breaker_rem_time = new_world.constants.wall_breaker_duration

        # update world with agent
        if is_us:
            new_world.agents[self.my_side] = current_agent
        else:
            new_world.agents[self.other_side] = current_agent

        return new_world

    def get_actions(self, world, player):
        actions = []
        cur_pos = player.position
        for direction in EDirection:
            if self.is_opposite_direction(direction, player.direction):
                continue
            if direction == EDirection.Up:
                # todo: check if we want to have Deliberate loss
                # todo: if index out of range error, might be here!
                ##
                y_destination = player.position.y - 1
                x_destination = player.position.x
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("up_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][x_destination] == ECell.BlueWall):
                            actions.append("up_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("up_on_enemy")
            elif direction == EDirection.Down:
                y_destination = player.position.y + 1
                x_destination = player.position.x
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("down_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][x_destination] == ECell.BlueWall):
                            actions.append("down_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("down_on_enemy")
            elif direction == EDirection.Left:
                y_destination = player.position.y
                x_destination = player.position.x - 1
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("left_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][x_destination] == ECell.BlueWall):
                            actions.append("left_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("left_on_enemy")
            elif direction == EDirection.Right:
                y_destination = player.position.y
                x_destination = player.position.x + 1
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("right_off")
                    else:
                        # hitting our wall
                        ####
                        if (self.my_side == "Yellow" and world.board[y_destination][x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][x_destination] == ECell.BlueWall):
                            actions.append("right_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("right_on_enemy")
        return actions

    def is_opposite_direction(self, direction1, direction2):
        if direction1 == EDirection.Up and direction2 == EDirection.Down:
            return True
        if direction1 == EDirection.Down and direction2 == EDirection.Up:
            return True
        if direction1 == EDirection.Left and direction2 == EDirection.Right:
            return True
        if direction1 == EDirection.Right and direction2 == EDirection.Left:
            return True
        return False

    def initialize(self):
        self.i = 1
        pass


    def decide(self):
        self.i += 1

        # depth = 1
        # self.min_max_tree(depth, self.world)

        # test
        best_move = None
        best_score = float('-inf')
        actions = self.get_actions(self.world, player=self.world.agents[self.my_side])
        if self.i >= 30:
            print("actions in i : " + str(self.i) + " are: " + actions.__str__())
        for action in actions:
            next_direction = action.split("_")[0]
            # activate_state = action.split("_")[1:]
            new_world = copy.deepcopy(self.world)
            new_world = self.game_result(new_world, next_direction, is_us=True)
            # new_score = self.min_val(depth, new_world)
            new_score = new_world.scores[self.my_side]
            if new_score > best_score:
                best_score = new_score
                best_move = action
                self.world = new_world

        split_move = best_move.split("_")[0]
        if split_move == "up":
            self.send_command(ChangeDirection(EDirection.Up))
        elif split_move == "down":
            self.send_command(ChangeDirection(EDirection.Down))
        elif split_move == "left":
            self.send_command(ChangeDirection(EDirection.Left))
        elif split_move == "right":
            self.send_command(ChangeDirection(EDirection.Right))

        print("our score : " + str(self.world.scores[self.my_side]))
        print("enemy score : " + str(self.world.scores[self.other_side]))
    # min max tree func
    def min_max_tree(self, depth, world):

        best_move = None
        best_score = float('-inf')
        for action in self.get_actions(world, player=self.my_side):
            next_direction = action.split("_")[0]
            activate_state = action.split("_")[1:]
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, next_direction, is_us=True)
            new_score = self.min_val(depth, new_world)
            if new_score > best_score:
                best_score = new_score
                best_move = action
                world = new_world
        # change agent direction to best move
        # todo: activate wall breaker if needed

        return best_score, best_move

    def min_val(self, depth, world):
        # if depth == 0:
        #     return self.heuristic(world)
        # v = float('inf')
        # deep copy world
        # min finding loop operation
        # modified world
        # return v

        # min is max of other player
        if depth == 0:
            return self.heuristic(world)
        v = float('-inf')
        depth -= 1
        for action in self.get_actions(world, player=self.my_side):
            # deep copy from world
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, action, player=self.other_side)
            v2 = self.max_val(depth, new_world)
            if v2 > v:
                v = v2
        return v

    def max_val(self, depth, world):
        if depth == 0:
            return self.heuristic(world)
        v = float('-inf')
        depth -= 1
        for action in self.get_actions(world, player=self.my_side):
            # deep copy from world
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, action, player=self.my_side)
            v2 = self.min_val(depth, new_world)
            if v2 > v:
                v = v2
        return v
