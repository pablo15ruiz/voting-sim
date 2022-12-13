import string

import numpy as np
import matplotlib.pyplot as plt


class Election:

    def __init__(self, n_vot, n_cand):
        self.n_vot = n_vot
        self.n_cand = n_cand

    def compute_winners(self, methods, util):
        winners = []
        poll = None
        if any([method.behaviour != 'honest' for method in methods]):
            poll = self.compute_poll(util)

        for method in methods:
            ballots = method.compute_ballots(util, poll)
            winner = method.compute_winner(ballots)
            winners.append(winner)

        return winners

    def compute_cycle(self, util):
        matrix = (util[:, None] < util[:, :, None]).astype(int).sum(axis=0)
        count = (matrix >= int(self.n_vot/2)+1).sum(axis=1)
        cycle = True if count.max() != self.n_cand-1 else False
        return cycle

    def compute_poll(self, util, threshold=0.5):
        return (util > threshold).sum(axis=0).argsort()

    def plot(self, voters, candidates):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.scatter(voters[:, 0], voters[:, 1], s=5)
        ax.scatter(candidates[:, 0], candidates[:, 1], s=70)
        names = list(string.ascii_uppercase)[:candidates.shape[0]]
        for i, txt in enumerate(names):
            x, y = candidates[i]
            ax.annotate(txt, (x, y))
        plt.show()


class Simulation:

    def __init__(self, n_vot, n_cand, n_iter, methods, voter_model):
        self.n_vot = n_vot
        self.n_cand = n_cand
        self.n_iter = n_iter
        self.methods = methods
        self.voter_model = voter_model

        self.vse = None
        self.cycle_freq = None

    def simulate(self):
        n_cycles = 0
        winners = []
        util_sum = []
        for _ in range(self.n_iter):
            util = self.voter_model.generate_util(self.n_vot, self.n_cand)
            util_sum.append(util.sum(axis=0))
            elec = Election(self.n_vot, self.n_cand)
            n_cycles += elec.compute_cycle(util)
            winners.append(elec.compute_winners(self.methods, util))

        self.cycle_freq = n_cycles/self.n_iter
        winners = np.array(winners)
        util_sum = np.array(util_sum)

        optimum = util_sum.max(axis=1).mean()
        idx = np.tile(range(self.n_iter), reps=(len(self.methods), 1)).T
        achieved = util_sum[idx, winners].mean(axis=0)
        average = util_sum.mean()
        self.vse = (achieved - average)/(optimum - average)

    def results(self):
        print(self)
        print(f'Condorcet cycle frequency: {np.round(100*self.cycle_freq, 3)}%')
        names = [str(method) for method in self.methods]
        d = dict(zip(names, np.round(100*self.vse, 2)))
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
        for name, metric in d.items():
            print(metric, name)

    def to_csv(self, filename='vse.csv'):
        header = ['method', 'behaviour', 'p', 'max_score', 'vse']
        data = [header]
        for method, value in zip(self.methods, self.vse):
            # name = metho
            data.append([method.name, method.behaviour, method.p, value])
        np.savetxt(filename, data, fmt='%s', delimiter = ",")


    def __str__(self):
        m = f'Simulation with {len(self.methods)} voting methods.\n'
        m += f'Voter model: {self.voter_model}\n'
        m += f'# Voters: {self.n_vot}\n'
        m += f'# Candidates: {self.n_cand}\n'
        m += f'# Iterations: {self.n_iter}\n'
        return m
