from abc import ABC, abstractmethod

import numpy as np
from numpy.random import choice, normal, random_sample, shuffle
from scipy.spatial import distance_matrix


class VoterModel(ABC):

    @abstractmethod
    def generate_util(self, n_vot, n_cand):
        pass

    def __repr__(self):
        return self.__class__.__name__


class ImpartialCulture(VoterModel):

    def generate_util(self, n_vot, n_cand):
        return random_sample((n_vot, n_cand))


class ClusteredSpatialModel(VoterModel):

    def generate_util(self, n_vot, n_cand):
        voters = self.generate_voters(n_vot)
        candidates = self.generate_candidates(voters, n_cand)
        util = self.compute_util(voters, candidates)
        return util

    def generate_voters(self, n_vot, n_dim=2):
        voters = np.empty((n_vot, n_dim))
        n_dim_per_view = self.chinese_restaurant_process(n_dim)
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

    def generate_candidates(self, voters, n_cand, n_dim=2):
        mean = voters.mean(axis=0)
        std = voters.std(axis=0)
        return 2*std*random_sample((n_cand, n_dim)) + mean - std

    def compute_util(self, voters, candidates):
        util = distance_matrix(voters, candidates)
        return 1 - util/util.max()

    @staticmethod
    def chinese_restaurant_process(n):
            n_tables = 1
            tables = np.array([1, 0])
            for customer in range(2, n+1):
                p = tables/customer
                p[-1] = 1/customer
                table_choice = choice(n_tables+1, p=p)
                tables[table_choice] += 1
                if table_choice == n_tables:
                    tables = np.append(tables, 0)
                    n_tables += 1

            return tables[:-1]
