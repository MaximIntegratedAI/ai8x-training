###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Evolutionary search for NAS
"""

import time

import numpy as np

from nas import nas_utils


class EvolutionSearch:
    """
    Evolutionary search for NAS
    """
    def __init__(self, population_size=100, prob_mutation=0.1, ratio_mutation=0.5,
                 ratio_parent=0.25, num_iter=500):
        self.population_size = population_size
        self.prob_mutation = prob_mutation
        self.ratio_mutation = ratio_mutation
        self.ratio_parent = ratio_parent
        self.num_iter = num_iter
        self.model = None
        self.arch = None

    def set_model(self, model):
        """Sets the trained base model"""
        self.model = model
        self.arch = model.get_base_arch()

    def set_model_arch(self, arch):
        """Sets the base model architecture"""
        self.arch = arch

    def get_random_valid_sample(self, constraint):
        """Randomly selects a valid sub network wrt the given constraint"""
        is_sample_ok = False
        while not is_sample_ok:
            sample = self.model.__class__.mutate(self.arch, self.arch, prob_mutation=1.0)
            is_sample_ok = self.check_constraint(sample, constraint)

        return sample

    def mutate_valid_sample(self, sample, constraint):
        """Mutates the sub network"""
        is_sample_ok = False
        while not is_sample_ok:
            new_sample = self.model.__class__.mutate(sample, self.arch, self.prob_mutation,
                                                     mutate_kernel=True, mutate_depth=True,
                                                     mutate_width=True)
            is_sample_ok = self.check_constraint(new_sample, constraint)

        return new_sample

    def crossover_valid_sample(self, sample1, sample2, constraint):
        """Crossovers two sub networks"""
        is_sample_ok = False
        while not is_sample_ok:
            new_sample = self.model.__class__.crossover(sample1, sample2)
            is_sample_ok = self.check_constraint(new_sample, constraint)

        return new_sample

    def check_constraint(self, sample, constraint):
        """Checks if the sub network meets the constarints"""
        if 'max_num_weights' in constraint:
            if self.model.__class__.get_num_weights(sample) > constraint['max_num_weights']:
                return False

        if 'min_num_weights' in constraint:
            if self.model.__class__.get_num_weights(sample) < constraint['min_num_weights']:
                return False

        if 'width_options' in constraint:
            unique_widths = self.model.__class__.get_unique_widths(sample)
            for width in unique_widths:
                if width not in constraint['width_options']:
                    return False

        return True

    def run(self, constraint, train_loader, test_loader, device):
        """Executes the search algorithm"""
        num_mutations = int(round(self.population_size * self.ratio_mutation))
        num_parents = int(round(self.population_size * self.ratio_parent))

        population = []
        best_acc = -9999
        best_arch = None

        print(f'Population Init with {self.population_size} Architecture Samples.')
        t1 = time.time()
        for _ in range(self.population_size):
            while True:
                child_net = self.get_random_valid_sample(constraint)
                if not nas_utils.check_net_in_population(child_net, population):
                    child_net_acc = nas_utils.calc_accuracy(child_net, self.model, train_loader,
                                                            test_loader, device)
                    child_net_eff = nas_utils.calc_efficiency(child_net)
                    break

            population.append((child_net, child_net_acc, child_net_eff))
        t2 = time.time()
        parents = sorted(population, key=lambda x: x[1], reverse=True)[:num_parents]
        best_acc = parents[0][1]
        best_arch = parents[0][0]
        print(f'\tBest Accuracy: {(100*best_acc):.2f}%')
        print(f'\tDuration: {(t2-t1):.2f}secs.\n')

        for n in range(self.num_iter):
            # Total popoulation size is equal to (population_size + num_parents)
            # after first iteration.
            t1_iter = time.time()
            print(f'Iteration: {n}')

            population = parents

            t1 = time.time()
            for _ in range(num_mutations):
                while True:
                    sample = population[np.random.randint(num_parents)][0]
                    child_net = self.mutate_valid_sample(sample, constraint)
                    if not nas_utils.check_net_in_population(child_net, population):
                        child_net_acc = nas_utils.calc_accuracy(child_net, self.model,
                                                                train_loader, test_loader, device)
                        child_net_eff = nas_utils.calc_efficiency(child_net)
                        break

                population.append((child_net, child_net_acc, child_net_eff))
            t2 = time.time()
            print(f'\tMutation done in {(t2-t1):.2f}secs.')

            t1 = time.time()
            for _ in range(self.population_size - num_mutations):
                while True:
                    sample1 = population[np.random.randint(num_parents)][0]
                    sample2 = population[np.random.randint(num_parents)][0]
                    child_net = self.crossover_valid_sample(sample1, sample2, constraint)
                    if not nas_utils.check_net_in_population(child_net, population):
                        child_net_acc = nas_utils.calc_accuracy(child_net, self.model,
                                                                train_loader, test_loader, device)
                        child_net_eff = nas_utils.calc_efficiency(child_net)
                        break

                population.append((child_net, child_net_acc, child_net_eff))
            t2 = time.time()
            print(f'\tCrossover done in {(t2-t1):.2f}secs.')

            parents = sorted(population, key=lambda x: x[1], reverse=True)[:num_parents]
            acc = parents[0][1]
            if acc > best_acc:
                best_arch = parents[0][0]
                best_acc = acc
            t2_iter = time.time()
            print(f'\tBest Accuracy: {(100*best_acc):.2f}%')
            print(f'\tBest Model: {best_arch}')
            print(f'\tDuration: {(t2_iter-t1_iter):.2f}secs.')

        # print('\nBest Models:')
        # for i in range(min(10, num_parents)):
        #     print(f'  Top-{i}')
        #     print('\tArch:', parents[i][0])
        #     print('\tArch:', parents[i][1])

        return parents
