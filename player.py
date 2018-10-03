#!/usr/bin/env python
from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d

from go_rules import Goban,Stone

BOARD_LEN = 9
np.random.seed(5)

# def timer(fun):
#     def fun_wrapper(*args,**kwargs):
#         t0 = time()
#         fun(*args,**kwargs)
#         print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
#     return fun_wrapper

def softmax(x):
    y = np.exp(x)
    return y/y.sum()

def play_game(players,display=False):
    goban = Goban()
    players[0].color = 'black'
    players[1].color = 'white'

    n_moves = 0
    turn = 0
    
    while not goban.is_over() and n_moves<200:
        move = players[turn].play(goban)
        goban.add_stone(move)
        n_moves += 1
        turn = 1-turn
        if display:
            goban.show()

    return players[1-turn]

class Player():
    def __init__(self,
                 ID,
                 color=None,
                 kernel_size=3):
        self.id = ID
        self.kernel_size = kernel_size
        self.neurons = 5
        self.color = color
        self.initialize()
        self.fitness = 0

    def initialize(self):
        self.matrices = [np.random.uniform(-1,1,[self.kernel_size]*2)
                       for _ in range(self.neurons)]
        self.weights = [np.random.uniform(0,1)
                        for _ in range(self.neurons)]
        self.normalize()

    def normalize(self):
        for matrix in self.matrices:
            matrix /= np.abs(matrix).sum()
        for weight in self.weights:
            weight /= np.sum(self.weights)

    def play(self,goban):
        goban_int = goban.to_int(self.color)
        pre_activations = [wi*convolve2d(goban_int,
                                         matrix,
                                         mode='same')
                           for (matrix,wi) in zip(self.matrices,self.weights)]
        pre_activations_w = np.sum(pre_activations,axis=0)
        
        activation = softmax(pre_activations_w)
        activation[goban_int!=0] = -1
        coord_max = np.unravel_index(activation.argmax(),activation.shape)
        return Stone(coord_max,self.color)

    def mutate(self,sd=0.01):
        for matrix in self.matrices:
            matrix += norm.rvs(0,sd,matrix.shape)
        self.weights += norm.rvs(0,sd,self.neurons)
        self.normalize()

if __name__ == '__main__':
    p1 = Player()
    p1.initialize()
    p2 = Player()
    p2.initialize()

    play_game([p1,p2],display=True)
    
