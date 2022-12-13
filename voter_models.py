from abc import ABC, abstractmethod

import numpy as np
from numpy.random import choice, normal, random_sample, shuffle
from scipy.spatial import distance_matrix
import random


class VoterModel(ABC):

    @abstractmethod
    def generate_util(self, n_vot, n_cand):
        pass

    def __str__(self):
        return self.__class__.__name__


class ImpartialCulture(VoterModel):

    def generate_util(self, n_vot, n_cand):
        return random_sample((n_vot, n_cand))


class SpatialModel(VoterModel):

    def __init__(self, n_dim=2):
        self.n_dim = n_dim

    def generate_util(self, n_vot, n_cand):
        voters = self.generate_voters(n_vot)
        candidates = self.generate_candidates(n_cand, voters)
        util = self.compute_util(voters, candidates)
        return util

    def generate_candidates(self, n_cand, voters):
        mean = voters.mean(axis=0)
        std = voters.std(axis=0)
        return 2*std*random_sample((n_cand, self.n_dim)) + mean - std

    def compute_util(self, voters, candidates):
        util = distance_matrix(voters, candidates)
        util = util.max() - util
        util = util / util.max()
        return util


class NormalSpatialModel(SpatialModel):

    def generate_voters(self, n_vot):
        return normal(0, 1, (n_vot, self.n_dim))


class ClusteredSpatialModel(SpatialModel):

    def generate_voters(self, n_vot):
        voters = np.empty((n_vot, self.n_dim))
        n_dim_per_view = self.chinese_restaurant_process(self.n_dim)
        views = np.repeat(range(len(n_dim_per_view)), n_dim_per_view)
        shuffle(views)
        for view_idx, d in enumerate(n_dim_per_view):
            n_vot_per_cluster = self.chinese_restaurant_process(n_vot)
            clusters = np.repeat(range(len(n_vot_per_cluster)), n_vot_per_cluster)
            shuffle(clusters)
            for cluster_idx, v in enumerate(n_vot_per_cluster):
                param = random_sample(2)
                cluster_voters = normal(param[0], param[1], (v, d))
                voter_idx = np.argwhere(clusters == cluster_idx)
                dim_idx = np.flatnonzero(views == view_idx)
                voters[voter_idx, dim_idx] = cluster_voters

        return voters

    @staticmethod
    def chinese_restaurant_process(n):
        tables = []
        while n:
            c = np.random.randint(1, n+1)
            tables.append(c)
            n -= c
        return tables
