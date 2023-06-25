# -*- coding: utf-8 -*-

# python imports
import random

# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)
        self.ver = EDirection.Down
        self.hor= EDirection.Right

    def rev_dir(self, cur):
        #dir = self.world.agents[self.my_side].direction
        dirs = [EDirection.Up, EDirection.Down, EDirection.Right, EDirection.Left]
        for i in range(4):
            if cur == dirs[i]:
                return dirs[i^1]
            
    def set_rev(self, cur):
        dirs = [EDirection.Up, EDirection.Down, EDirection.Right, EDirection.Left]
        for i in range(4):
            if cur == dirs[i]:
                if i<2:
                    self.ver= dirs[i^1]
                else:
                    self.hor= dirs[i^1]
                break

    def will_hit(self, dir, hit_area=False):
        p = self.world.agents[self.my_side].position
        dx, dy= 0, 0
        if dir == EDirection.Down:
            dy = +1
        if dir == EDirection.Up:
            dy = -1
        if dir == EDirection.Left:
            dx = -1
        if dir == EDirection.Right:
            dx = 1
        
        if p.x+dx < 0 or p.x+dx >= len(self.world.board[0]) or p.y+dy < 0 or p.y+dy>= len(self.world.board):
            print("out of", p.x, p.y, dir, p.x+dx, p.y+dy, "len", len(self.world.board[0]), len(self.world.board))
            return True
        if not hit_area:
            if self.world.board[p.y+dy][p.x+dx] != ECell.Empty:
                print("in hit:", p.x, p.y, dir, p.x+dx, p.y+dy, self.world.board[p.y+dy][p.x+dx])
                return True
        if hit_area:
            if self.world.board[p.y+dy][p.x+dx] == ECell.AreaWall:
                print("in hit:", p.x, p.y, dir, p.x+dx, p.y+dy, self.world.board[p.y+dy][p.x+dx])
                return True
        return False
    
    def rot(self, dir, x):
        dirs = [EDirection.Up, EDirection.Right, EDirection.Down, EDirection.Left]
        for i in range(4):
            if dir == dirs[i]:
                return dirs[(i+x+4)%4]

    def next_move(self):
        dir = self.world.agents[self.my_side].direction
        p = self.world.agents[self.my_side].position
        print("cur dir and pos", dir, p.x, p.y)
        next = None
        if dir == self.ver:
            next= self.hor
        elif dir == self.hor:
            next= self.ver
        else:
            if not self.will_hit(self.rot(dir, 1)):
                self.send_command(ChangeDirection(self.rot(dir, 1)))
            elif not self.will_hit(self.rot(dir, -1)):
                self.send_command(ChangeDirection(self.rot(dir, -1)))
            else:
                #self.hor, self.ver = self.rev_dir(self.hor), self.rev_dir(self.ver)
                self.send_command(ChangeDirection(dir))
            return

        print("next is", next)
        
        if self.will_hit(next):
            print("will hit")
            if not self.will_hit(self.rev_dir(next)):
                self.set_rev(next)
                self.send_command(ChangeDirection(self.rev_dir(next)))
            elif not self.will_hit(dir):
                self.send_command(ChangeDirection(dir))
            else:
                not_area = next
                if self.will_hit(not_area, True):
                    not_area = self.rev_dir(next)
                    if not self.will_hit(not_area, True):
                        self.set_rev(next)
                    else:
                        not_area= dir

                if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
                    self.send_command(ActivateWallBreaker())
                self.send_command(ChangeDirection(not_area))
        else:
            self.send_command(ChangeDirection(next))
        
    #def choose_zig(self):
    #    p = self.world.agents[self.my_side].position

    def initialize(self):
        print('initialize')
        p = self.world.agents[self.my_side].position
        self.ver = EDirection.Down
        self.hor= EDirection.Right
        dir = self.world.agents[self.my_side].direction
        #if self.will_hit(dir):
        #if dir != "Down" and dir != EDirection.Right:
        #    self.ver = EDirection.Up
        #    self.hor= EDirection.Left

        if self.will_hit(self.ver):
            self.ver = EDirection.Up
        if self.will_hit(self.hor):
            self.hor = EDirection.Left
        print(p.x, p.y, self.hor, self.ver, self.my_side, dir)
        print("sizes:", len(self.world.board[0]), len(self.world.board))


    def decide(self):
        print('decide')
        self.next_move()
