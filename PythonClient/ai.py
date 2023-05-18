# -*- coding: utf-8 -*-

# python imports
import random
from torch import nn


# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker


class MagicalBrain(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 
        
        
        
    def forward(self):
        pass 


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)


    def initialize(self, brain: MagicalBrain):
        self.brain = brain


    def decide(self):
        #bounded depth MinMax Tree Search
        #calculating heuristics by using the our MagicalBrain
        #choosing the best move
        
        self.send_command()
        if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
            self.send_command(ActivateWallBreaker())
                 
    def world_as_tensor(self):
        pass


class Genome:
    def __init__(self, AI):
        self.brain = AI.brain
        pass
    
    def genome(self):
        # make a genome from the brain
        pass 
    
    def mutate(self):
        # mutate the genome
        pass
    
    def crossover(self, other_genome):
        # crossover with other genome
        pass
    
    def get_brain(self):
        # return the brain
        pass