from ai import AI, Soul
import torch
from torch import nn


#demons fight to reach the apex
class FightClub:
    def __init__(self):
        self.demons = []
        
    
class Demon(AI):
    def __init__(self, soul = None) -> None:
        super().__init__()
        self.parmeters = {'hidden_layers': [], 'input_size': 0}
        self.soul = Soul(**self.parmeters) if soul is None else soul
    
    @staticmethod
    def sex(demon1, demon2):
        soul1 = demon1.soul
        soul2 = demon2.soul
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

        new_soul1 = Soul(hidden_layers=soul1.hidden_layers, input_size=soul1.input_size)
        new_soul2 = Soul(hidden_layers=soul1.hidden_layers, input_size=soul1.input_size)

        for i, param in enumerate(new_soul1.parameters()):
            param = nn.parameter.Parameter(new_wights1[i])
        for i, param in enumerate(new_soul2.parameters()):
            param = nn.parameter.Parameter(new_wights2[i])
        
        return (Demon(new_soul1), Demon(new_soul2))