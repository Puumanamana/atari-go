#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
from itertools import chain,product
import numpy as np

BOARD_LEN = 9

class Stone:
    def __init__(self,coords,color):
        self.coords = coords
        self.group = None
        self.color = color

    def neighbors(self,state,color=None,invert=False):
        neighbors = []
        i,j = self.coords

        intersections = [[i,i,i+1,i-1],[j+1,j-1,j,j]]
            
        for k,l in zip([i,i,i+1,i-1],[j+1,j-1,j,j]):
            try:
                if state[k,l] is not None and not invert:
                    if color is None:
                        neighbors.append(state[k,l])
                    else:
                        if state[k,l].color == color:
                            neighbors.append(state[k,l])
                elif state[k,l] is None and invert: # we are counting liberties
                    neighbors.append((k,l))
                            
            except KeyError:
                pass
                
        return neighbors
    
class Goban():
    def __init__(self):
        self.state = { (i,j): None
                       for (i,j) in product(
                               range(BOARD_LEN),range(BOARD_LEN),repeat=True)
        }
        self.group_libs = {}
        self.initialize()

    def initialize(self):
        self.add_stone(Stone((4,4),'black'))
        self.add_stone(Stone((5,5),'black'))        
        self.add_stone(Stone((4,5),'white'))
        self.add_stone(Stone((5,4),'white'))        

    def get_stones(self):
        return [stone for stone in self.state.values()
                if stone is not None]
        
    def to_int(self,color):
        board = np.zeros([BOARD_LEN]*2)
        for coord,stone in self.state.items():
            if stone is not None:
                board[coord] = 2*int(stone.color==color)-1
        return board
        
    # Merge groupe g1 and g2 and compute new liberties
    def merge_groups(self,g1,g2):
        for stone in self.get_stones():
            if stone.group == g2:
                stone.group = g1
        del self.group_libs[g2]
        self.count_liberties(g1)

    # Count liberties in group
    def count_liberties(self,group):
        stones = [ stone_g for stone_g
                   in self.get_stones()
                   if stone_g.group == group ]
        
        neighs = chain(*[stone.neighbors(self.state,
                                         color=stone.color,
                                         invert=True)
                         for stone in stones])
        
        self.group_libs[group] = len(set(neighs))
        
    # Add stone on goban and creates new group/call merging if needed
    def add_stone(self,stone):
        self.state[stone.coords] = stone

        neighbors = stone.neighbors(self.state)
        N_common = 0

        for neigh in neighbors:
            # Stone added to a new group
            if neigh.color == stone.color and N_common == 0:
                N_common += 1
                stone.group = neigh.group
                self.count_liberties(neigh.group)
                
            # Stone links 2 groups
            if neigh.color == stone.color and \
               stone.group != neigh.group and \
                              N_common == 1:
                self.merge_groups(stone.group,neigh.group)
                N_common = 1

            # Stone touch ennemy group
            if neigh.color != stone.color:
                self.group_libs[neigh.group] -= 1
                
        # Stone has no friends -> new group
        if N_common == 0:
            if len(self.group_libs)>0:
                stone.group = np.max(list(self.group_libs.keys())) + 1
            else:
                stone.group = 0
            
            self.count_liberties(stone.group)
                
    def is_over(self):
        return (0 in self.group_libs.values())

    def show(self,display_time=2):
    
        fig,ax = plt.subplots()
        ax.set_facecolor((0.66, 0.34, 0))
        ax.set_xlim([0,8])
        ax.set_ylim([0,8])

        stones = self.get_stones()

        points = [stone.coords for stone in stones]
        colors = ['k' if stone.color == 'black' else 'w'
                  for stone in stones]
        
        plt.scatter(*zip(*points),c=colors,s=800)
        for stone in stones:
            col = 'w'*(stone.color=='black')+'k'*(stone.color=='white')
            ax.annotate(self.group_libs[stone.group],
                        stone.coords,
                        color=col,
                        fontsize=12)
        plt.show(block=False)
        plt.grid(True)
        plt.pause(display_time)
        plt.close()
    
if __name__ == '__main__':
    game = Goban()
    game.show()
