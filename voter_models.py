from abc import ABC, abstractmethod

from numpy.random import random_sample

class VoterModel(ABC):

    @abstractmethod
    def generate_utilities(self, n_vot, n_cand):
        pass

class ImpartialCulture(VoterModel):

    def generate_utilities(self, n_vot, n_cand):
        return random_sample((n_vot, n_cand))


class ClusteredSpatialModel(VoterModel):

    def generate_utilities(self, n_vot, n_cand):
        voters = self.generate_voters(n_vot, n_dim=5)
        candidates = self.generate_candidates(n_cand)
        utilities = distance_matrix(voters, candidates)
        return 1 - utilities/utilities.max()

    def generate_voters(self, n_vot, n_dim):
        voters = np.empty((n_vot, n_dim))
        n_dim_per_view = chinese_restaurant_process(n_dim)
        views = np.repeat(range(len(n_dim_per_view)), n_dim_per_view)
        np.random.shuffle(views)
        for view_idx, n_dim in enumerate(n_dim_per_view):
            n_voters_per_cluster = chinese_restaurant_process(n_vot)
            clusters = np.repeat(range(len(n_voters_per_cluster)), n_voters_per_cluster)
            np.random.shuffle(clusters)
            for cluster_idx, n_voters in enumerate(n_voters_per_cluster):
                for dim in range(n_dim):
                    parameters = np.random.rand(2)
                    cluster_voters = np.random.normal(
                        parameters[0],
                        parameters[1],
                        n_voters
                    )
                    voter_idx = [idx for idx, c in enumerate(clusters) if c == cluster_idx]
                    dim_idx = [idx for idx, v in enumerate(views) if v == view_idx][dim]
                    voters[voter_idx, dim_idx] = cluster_voters

    def generate_candidates(self, n_cand, n_dim):
        mean = voters.mean(axis=0)
        std = voters.std(axis=0)
        candidates = np.random.random_sample(
            (self.n_candidates, n_dim)
        )
        candidates = 2*std*candidates + mean - std

    @staticmethod
    def chinese_restaurant_process(n):
            n_tables = 1
            tables = np.array([1, 0])
            for customer in range(2, n+1):
                p = tables/customer
                p[-1] = 1/customer
                table_choice = np.random.choice(n_tables+1, p=p)
                tables[table_choice] += 1
                if table_choice == n_tables:
                    tables = np.append(tables, 0)
                    n_tables += 1

            return tables[:-1]
