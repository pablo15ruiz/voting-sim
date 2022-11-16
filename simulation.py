import numpy as np
from numpy.random import choice

class Election:

    def __init__(self, n_vot, n_cand, utilities, poll=False):
        self.n_vot = n_vot
        self.n_cand = n_cand
        self.utilities = utilities
        self.poll = self.do_poll() if poll else None

    def compute_winner(self, method):
        poll = self.poll if method.behaviour != 'honest' else None
        ballots = method.compute_ballots(self.utilities, poll)
        winner = method.compute_winner(ballots)
        if winner == None:
            winner = choice(range(self.n_cand))

        return winner

    def do_poll(self, threshold=0.7):
        return (self.utilities > threshold).sum(axis=0)

    def plot(self, voters, candidates):
        pass


class Simulation:

    def __init__(self, n_vot, n_cand, n_iter, methods, voter_model):
        self.n_vot = n_vot
        self.n_cand = n_cand
        self.n_iter = n_iter
        self.methods = methods
        self.voter_model = voter_model
        self.poll = any([method.behaviour != 'honest' for method in methods])

        self.total_utilities = []
        self.total_winners = []

    def simulate(self):
        winners = []
        for _ in range(self.n_iter):
            utilities = self.voter_model.generate_utilities(
                self.n_vot,
                self.n_cand
            )
            self.total_utilities.append(utilities.sum(axis=0))
            election = Election(
                self.n_vot,
                self.n_cand,
                utilities,
                poll=self.poll
            )
            election_winners = []
            for method in self.methods:
                winner = election.compute_winner(method)
                election_winners.append(winner)
            winners.append(election_winners)

        self.total_winners = np.array(winners)
        self.total_utilities = np.array(self.total_utilities)

    def compute_metrics(self):
        optimum = self.total_utilities.max(axis=1).mean()
        idx = np.tile(range(self.n_iter), reps=(len(self.methods), 1)).T
        achieved = self.total_utilities[idx, self.total_winners].mean(axis=0)
        average = self.total_utilities.mean()
        self.vse = (achieved - average)/(optimum - average)

    def show_results(self):
        names = [f'{method} {method.behaviour}' for method in self.methods]
        d = dict(zip(names, np.round(100*self.vse, 2)))
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
        for name, metric in d.items():
            print(metric, name)
