from PythonClient.ai import AI, Soul
import torch
from torch import nn
from PythonClient.ks.models import ECell, EDirection, Position
import numpy as np


# demons fight to reach the apex
class FightClub:
    def __init__(self):
        self.demons = []

    # demons fight every day
    def day(self):
        pass

        # demons sex every night

    def night(self):
        pass

    def fight(demon1, demon2):
        # rand.choice(maps)
        pass

    # you don't know How heartless they are


# They don't have any feelings
# They are born to fight
class Demon(AI):
    def __init__(self, soul=None) -> None:
        super().__init__()
        self.parmeters = {'hidden_layers': [], 'input_size': 0}
        self.soul = Soul(**self.parmeters) if soul is None else soul

    # they make twin as soon as they sex =)
    @staticmethod
    def sex(mother, father):
        soul1 = mother.soul
        soul2 = father.soul

        rands = []
        rand_comps = []
        new_wights1 = []
        new_wights2 = []

        for param in soul1.parameters():
            rand = torch.rand_like(param)
            rands.append(rand)
            rand_comp = 1 - rand
            rand_comps.append(rand_comp)
            new_wights1.append(rand * param)
            new_wights2.append(rand_comp * param)

        for i, param in enumerate(soul2.parameters()):
            new_wights1[i] += rand_comps[i] * param
            new_wights2[i] += rand[i] * param

        son_soul = Soul(hidden_layers=soul1.hidden_layers, input_size=soul1.input_size)
        daughter_soul = Soul(hidden_layers=soul1.hidden_layers, input_size=soul1.input_size)

        for i, param in enumerate(son_soul.parameters()):
            param = nn.parameter.Parameter(new_wights1[i])
        for i, param in enumerate(daughter_soul.parameters()):
            param = nn.parameter.Parameter(new_wights2[i])

        return Demon(son_soul), Demon(daughter_soul)

    def get_ready_for_fight(self, world):
        self.world = world

    def inspect(self):
        if self.world is None:
            world = self.world
        else:
            return None
        position = self.world.agents[self.my_side].position
        # get 3 * 3 block which my agent is mid and down
        neighbors = self.world.board[position[0] - 1:position[0] + 2][ position[1] - 1:position[1] + 2]

        for i in range(3):
            for j in range(3):
                if neighbors[i][j] == ECell.Empty:
                    neighbors[i][j] = [1, 0, 0, 0]
                elif neighbors[i][j] == ECell.AreaWall:
                    neighbors[i][j] = [0, 1, 0, 0]
                elif self.my_side == 'Blue' and neighbors[i][j] == ECell.BlueWall \
                        or self.my_side == 'Yellow' and neighbors[i][j] == ECell.YellowWall:
                    neighbors[i][j] = [0, 0, 1, 0]
                else:
                    neighbors[i][j] = [0, 0, 0, 1]

        # make a numpy array from neighbors
        neighbors = np.array(neighbors)
        # rotate neighbors to match the agent direction
        neighbors = np.rot90(neighbors, self.world.agents[self.my_side].direction.value)
        # make a 1D array from neighbors
        neighbors = neighbors.flatten()
        # concatenating other information
        my_side_info = [self.world.agents[self.my_side].wall_breaker_cooldown,
                        self.world.agents[self.my_side].wall_breaker_rem_time,
                        self.world.agents[self.my_side].position[0],
                        self.world.agents[self.my_side].position[1],
                        self.world.scores[self.my_side]]
        informations = np.concatenate((neighbors,
                                       ))
        # concatenating other agents information
        np.concatenate((neighbors,
                        [self.world.agents[self.other_side].wall_breaker_cooldown,
                         self.world.agents[self.other_side].wall_breaker_rem_time,
                         self.world.agents[self.other_side].position[0],
                         self.world.agents[self.other_side].position[1],
                         self.world.agents[self.other_side].score],))
