#!/usr/bin/env python

from player import Player
from player import play_game

from multiprocessing.pool import Pool
from itertools import combinations
from time import time

import numpy as np
from scipy.stats import cauchy,norm

import matplotlib.pyplot as plt

np.random.seed(1234)

def timer(fun):
    def fun_wrapper(*args,**kwargs):
        t0 = time()
        fun(*args,**kwargs)
        print('{0} - Elapsed time: {1:.3f}'.format(fun.__name__,time()-t0))
    return fun_wrapper

class Evolution:
      
    def __init__(self,N):
        self.pop_size = N
        self.populations = {i: Player(i) for i in range(N)}
        self.recombine_prob = 50./N
        self.mutation_rate = 10./N
        self.cores = 5
        self.fitnesses = []

    def reset_fitnesses(self):
        for p in self.populations.values():
            p.fitness = 0

    @timer
    def calc_fitnesses(self,max_games=50):
        self.reset_fitnesses()
        pool = Pool(5)
        winners = pool.map(play_game,
                           combinations(self.populations.values(),2))
        for winner in winners:
            self.populations[winner.id].fitness += 1

        self.fitnesses.append([p.fitness for p in self.populations.values()])
        print(sorted(self.fitnesses[-1],reverse=True))
        

    @timer
    def mutate_generation(self):
        for player in self.populations.values():
            if np.random.uniform() < self.mutation_rate:
                player.mutate(sd=0.1)

    def recombine(self,players):
        fitnesses = [p.fitness for p in players]
        choices = np.random.choice(2,players[0].neurons,
                                   p=fitnesses/np.sum(fitnesses))
        child = Player(-1)
        child.matrices = [ players[choice].matrices[i]
                           for (i,choice) in enumerate(choices)]
        child.weights = [ players[choice].weights[i]
                          for (i,choice) in enumerate(choices)]
        child.normalize()
        return child

    def select_next_gen(self):
        fitnesses = np.array([ p.fitness for p in self.populations.values() ])
        sorted_fitnesses = sorted([(i,p) for (i,p) in enumerate(fitnesses)],
                                  key=lambda x: x[1])

        recombination_probs = fitnesses / np.sum(fitnesses)

        parents = np.random.choice(range(self.pop_size),
                                   int(self.recombine_prob*self.pop_size),
                                   p=recombination_probs,
                                   replace=False)        
        removed = [ i for i,f in sorted_fitnesses if i not in parents ]
    
        return parents,removed[0:min(len(parents),len(removed))]

    @timer
    def make_next_generation(self):
        parents,removed = self.select_next_gen()

        couples = list(combinations(parents,2))
        indices = np.random.choice(len(couples),len(removed),replace=False)

        n_children = 0
        for ids in [couples[i] for i in indices]:
            if np.random.uniform() < self.recombine_prob:
                parents = [self.populations[n] for n in ids]
                child = self.recombine(parents)
                child.id = removed[n_children]
                self.populations[removed[n_children]] = child
                n_children += 1

    def cycle(self):
        self.calc_fitnesses()
        self.make_next_generation()
        self.mutate_generation()

    def cycles(self,n_gen):
        for n in range(n_gen):
            print('Generation {}/{}'.format(n,n_gen))
            self.cycle()
        self.demo()
        # self.display_fitness()
        return self

    def demo(self,ax=None,indices=[0,1]):
        fitnesses = np.array([ p.fitness for p in self.populations.values() ])
        sorted_fitnesses = sorted([(i,p) for (i,p) in enumerate(fitnesses)],
                                  key=lambda x: x[1],
                                  reverse=True)
        best_players = [self.populations[sorted_fitnesses[indices[0]][0]],
                        self.populations[sorted_fitnesses[indices[1]][0]]]
        play_game(best_players,display=True)

    def display_fitness(self):
        fig,ax = plt.subplots()
        for i,v in enumerate(ev.fitnesses):
            ax.scatter([i]*len(v),v,s=1,c='b')
        ax.plot(range(len(ev.fitnesses)),list(map(np.mean,ev.fitnesses)),label='mean',c='k')
        ax.plot(range(len(ev.fitnesses)),list(map(np.max,ev.fitnesses)),label='max',c='g')
        plt.legend()
        plt.show()
        
    
if __name__ == '__main__':

    ev = Evolution(100)
    ev.cycles(5000)

    
