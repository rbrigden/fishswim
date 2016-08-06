# -*- coding: utf-8 -*-
# this GA teaches fish to swim expertly
# the ocean is a 2D array. 1 (seaweed) and 0 (open water)

import random
import yaml
from datetime import datetime
import math
import logging
import numpy as np
import sys
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(filename='./log/'+str(datetime.now().isoformat())+'.log',level=logging.DEBUG)
config = yaml.safe_load(open("./config.yml"))

OCEANSIZE = config["oceansize"]
START = (config["startx"],config["starty"])
END = (OCEANSIZE-1, OCEANSIZE-1)
POP_SIZE    = config["popsize"]
GENERATIONS = config["generations"]
MUTATION_CHANCE = config["mutation_chance"]
DATA_PATH = config["data_path"]


class Fish:

    def __init__(self, ocean):
        self.ocean = ocean
        self.path = []
        self.collisions = 0
        self.x, self.y = START
        self.swam = False

    # let the fish swim
    def drop_in_ocean(self):
        if not self.swam:
            self.path.append(START)
            while (self.x, self.y) != END:
                if self.move(random.randrange(0,8)):
                    self.path.append( (self.x, self.y) )


    # for each move, the distance is shrinking to the end
    # if there is seaweed neglect the contribution or double the negation
    def fitness(self):
        result = 0
        for idx in xrange (len(self.path)-1):
            cpos = self.path[idx]
            npos = self.path[idx]
            delta = distance_to_end(cpos) - distance_to_end(npos)
            if delta < 0:
                # add to result if we move away
                result += abs(delta)
            result += 1000 * self.collisions
        return result

    def path_length(self):
        return len(self.path)

    def seaweed(self):
        result = 0
        for pos in self.path:
            col, row = pos
            if self.ocean[row][col] == 1:
                result += 1
        return result

    # return pos after move
    # 0 1 2
    # 3 x 4
    # 5 6 7
    def move(self, direction):
        # (col, row)
        moves = [(-1,-1), (0,-1), (1,-1),(-1, 0), (1, 0), (-1, 1), (0, 1), (1,1)]
        move = moves[direction]
        if self.legal_move(move):
            self.x += move[0]
            self.y += move[1]
            if self.ocean[move[1]][move[0]] == 1:
                self.collisions += 1
        else:
            return False
        return True

    def get_next_move(self, direction):
        # (col, row)
        moves = [(-1,-1), (0,-1), (1,-1),(-1, 0), (1, 0), (-1, 1), (0, 1), (1,1)]
        move = moves[direction]
        if self.legal_move(move):
            return (self.x + move[0], self.y + move[1])
        return (self.x, self.y)

    def legal_move(self, move):
        new_x = self.x + move[0]
        new_y = self.y + move[1]
        x_in_bound = 0 <= new_x and new_x < OCEANSIZE
        y_in_bound = 0 <= new_y and new_y < OCEANSIZE
        if x_in_bound and y_in_bound:
            return True
        else:
            return False

    def mutate(self):
        """
        For each path_length in a fishes path, there is a 1/mutation_chance chance that it will be
        moved 1 space while maintaining path integrity. This ensures diversity in the
        population, and ensures that is difficult to get stuck in local minima.
        """
        fish_out = copy.copy(self)
        for idx in xrange(1, len(fish_out.path)-1):
            self.x, self.y = fish_out.path[idx]
            if int(random.random()*MUTATION_CHANCE) == 1:
                back = fish_out.path[idx-1]
                forward = fish_out.path[idx+1]
                self.x, self.y = fish_out.path[idx]
                new_pos = self.get_next_move(random.randrange(0,8))
                if (is_adjacent(new_pos, back) and is_adjacent(new_pos, forward)):
                    fish_out.path[idx] = new_pos
        assert(is_continuous_path(fish_out.path))
        return fish_out



    def print_solution(self):
        temp = list(self.ocean)
        idx = 0
        for pos in self.path:
            print pos
            temp[pos[1]][pos[0]] = str(unichr(ord('a')+(idx%26)))
            idx+=1
        print("solution")
        print(np.matrix(temp))

def crossover(fish1, fish2):
    """
    Find all intersections of the paths of fish1 and fish 2 and pick a
    random intersection at which to splice the paths. Then create a hybrid
    continuous path for a fish of the next generation.
    """

    intersections = set(fish1.path).intersection(fish2.path)
    intersection = random.choice(list(intersections))
    f1_indeces =  [i for i,x in enumerate(fish1.path) if x == intersection]
    f2_indeces =  [i for i,x in enumerate(fish2.path) if x == intersection]
    new_f1_path = fish1.path[:random.choice(f1_indeces)] + fish2.path[random.choice(f2_indeces):]
    new_f2_path = fish2.path[:random.choice(f2_indeces)] + fish1.path[random.choice(f1_indeces):]
    new_f1 = copy.copy(fish1)
    new_f2 = copy.copy(fish2)
    new_f1.path = new_f1_path
    new_f2.path = new_f2_path

    assert(is_continuous_path(new_f1_path))
    assert(is_continuous_path(new_f2_path))

    return new_f1, new_f2


def is_continuous_path(path):
    for idx in xrange (len(path)-1):
        cpos = path[idx]
        npos = path[idx+1]
        if not is_adjacent(cpos, npos):
            return False
    return True

def is_adjacent(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    if abs(x1-x2) == 1 and abs(y1-y2) == 0 or abs(x1-x2) == 0 and abs(y1-y2) == 1:
        return True
    if abs(x1-x2) == 1 and abs(y1-y2) == 1:
        return True
    return False

def distance_to_end(pos):
    endx = END[0]
    endy = END[1]
    return math.sqrt((pos[0]-endx)**2 + (pos[1]-endy)**2)

# builds ocean OCEANSIZE x OCEANSIZE
def build_ocean():
    ocean = []
    for row in xrange(OCEANSIZE):
        ocean.append([random.randrange(0,2)*random.randrange(0,2) for _ in range (OCEANSIZE)])
    return ocean

def build_population(ocean):
    population = []
    for i in range(POP_SIZE):
        fish = Fish(ocean)
        fish.drop_in_ocean()
        population.append(fish)
    return population


def weighted_choice(items):
    """
    Chooses a random element from items, where items is a list of tuples in
    the form (item, weight). weight determines the probability of choosing its
    respective item. Note: this function is borrowed from ActiveState Recipes.
    """
    weight_total = sum((item[1] for item in items))
    n = random.uniform(0, weight_total)
    for item, weight in items:
        if n < weight:
            return item
        n = n - weight
    return item


def evolve():
    random.seed(134544)
    ocean = build_ocean()
    population = build_population(ocean)
    for generation in xrange(GENERATIONS):
        msg = "Generation %s... Random sample, Path Length: %d Collisions: %d" % (generation, population[0].path_length(), population[0].seaweed())
        print(msg)
        logging.info(msg)
        weighted_population = []

        for individual in population:
            fitness_val = individual.fitness()

        if fitness_val == 0:
            pair = (individual, 1.0)
        else:
            pair = (individual, 1.0/fitness_val)

        weighted_population.append(pair)

        population = []

        for _ in xrange(POP_SIZE/2):
            # Selection
            ind1 = weighted_choice(weighted_population)
            ind2 = weighted_choice(weighted_population)

            # Crossover
            ind1, ind2 = crossover(ind1, ind2)

            # Mutate and add back into the population.
            population.append(ind1.mutate())
            population.append(ind2.mutate())

    fittest_fish = population[0]
    minimum_fitness = population[0].fitness()

    for individual in population:
        ind_fitness = individual.fitness()
        if ind_fitness <= minimum_fitness:
                fittest_fish = individual
                minimum_fitness = ind_fitness
    return fittest_fish
