import string
import numpy as np

class Election:

    def __init__(self, n_vot, n_cand, util, poll=False):
        self.n_vot = n_vot
        self.n_cand = n_cand
        self.util = util
        self.poll = self.do_poll() if poll else None
        self.cw = self.compute_condorcet_winner()

    def compute_winner(self, method):
        poll = self.poll if method.behaviour != 'honest' else None
        ballots = method.compute_ballots(self.util, poll)
        winner = method.compute_winner(ballots)
        return winner

    def compute_condorcet_winner(self):
        # print(np.round(self.util, 3))
        matrix = (self.util[:, None] < self.util[:, :, None]).astype(int).sum(axis=0)
        count = (matrix >= int(self.n_vot/2)+1).sum(axis=1)
        winner = count.argmax() if count.max() == self.n_cand-1 else None
        return winner

    def do_poll(self, threshold=0.7):
        return (self.util > threshold).sum(axis=0).argsort()

    def plot(self, voters, candidates):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
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

        self.total_util = []
        self.total_winners = []
        self.cycle_freq = None

    def simulate(self):
        n_cycles = 0
        winners = []
        poll = any([method.behaviour != 'honest' for method in self.methods])
        for _ in range(self.n_iter):
            util = self.voter_model.generate_util(self.n_vot, self.n_cand)
            self.total_util.append(util.sum(axis=0))
            elec = Election(self.n_vot, self.n_cand, util, poll=poll)
            n_cycles += elec.cw is None
            w = [elec.compute_winner(method) for method in self.methods]
            winners.append(w)

        self.total_winners = np.array(winners)
        self.total_util = np.array(self.total_util)
        self.cycle_freq = n_cycles/self.n_iter

        self.compute_metrics()

    def compute_metrics(self):
        optimum = self.total_util.max(axis=1).mean()
        idx = np.tile(range(self.n_iter), reps=(len(self.methods), 1)).T
        achieved = self.total_util[idx, self.total_winners].mean(axis=0)
        average = self.total_util.mean()
        self.vse = (achieved - average)/(optimum - average)

    def results(self):
        print(self)
        print(f'Condorcet cycle frequency: {np.round(100*self.cycle_freq, 3)}%')
        names = [str(method) for method in self.methods]
        d = dict(zip(names, np.round(100*self.vse, 2)))
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
        for name, metric in d.items():
            print(metric, name)

    def to_csv(self, filename='data.csv'):
        '''method, behaviour, p, vse'''
        header = ['method', 'behaviour', 'p', 'vse']
        data = [header]
        for method, value in zip(self.methods, self.vse):
            data.append([method.name, method.behaviour, method.p, value])
        np.savetxt(filename, data, fmt='%s', delimiter = ",")


    def __str__(self):
        m = f'Simulation with {len(self.methods)} voting methods.\n'
        m += f'Voter model: {self.voter_model}\n'
        m += f'#Voters: {self.n_vot}\n'
        m += f'#Candidates: {self.n_cand}\n'
        m += f'#Iterations: {self.n_iter}\n'
        return m
