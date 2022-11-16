from abc import ABC, abstractmethod

import numpy as np

class VotingMethod(ABC):

    def __init__(self, behaviour):
        behaviour_method = f'compute_ballots_{behaviour}'
        if behaviour_method not in dir(self):
            raise NameError(f'Method ({self} {behaviour}) not implemented')

        self.behaviour = behaviour
        self.compute_ballots = getattr(self, behaviour_method)

    @abstractmethod
    def compute_winner(self, ballots):
        pass

    def __repr__(self):
        return self.__class__.__name__

class FPTP(VotingMethod):

    def __init__(self, behaviour):
        super().__init__(behaviour)

    def compute_winner(self, ballots):
        return np.bincount(ballots).argmax()

    def compute_ballots_honest(self, utilities, poll=None):
        return utilities.argmax(axis=1)

    def compute_ballots_strategy(self, utilities, poll=None):
        front_runners = np.argsort(poll)[-2:]
        return front_runners[utilities[:, front_runners].argmax(axis=1)]

class Borda(VotingMethod):

    def __init__(self, behaviour):
        super().__init__(behaviour)

    def compute_winner(self, ballots):
        return ballots.sum(axis=0).argmin()

    def compute_ballots_honest(self, utilities, poll=None):
        return np.flip(utilities.argsort(), axis=1).argsort()

class Condorcet(VotingMethod):

    def __init__(self, behaviour):
        super().__init__(behaviour)

    def compute_winner(self, ballots):
        n_voters = len(ballots)
        matrix = (ballots[:, None] > ballots[:, :, None]).astype(int).sum(axis=0)
        count = (matrix >= int(n_voters/2)).sum(axis=1)
        winner = np.random.choice(range(len(count)))
        if count.max() == len(count) - 1:
            winner = count.argmax()
        return winner

    def compute_ballots_honest(self, utilities, poll=None):
        return np.flip(utilities.argsort(), axis=1).argsort()

class IRV(VotingMethod):

    def __init__(self, behaviour):
        super().__init__(behaviour)

    def compute_winner(self, ballots):
        candidates = [*range(ballots.shape[1])]
        while len(candidates) > 1:
            worst = np.bincount(
                ballots.argmin(axis=1),
                minlength=len(candidates)
            ).argmin()
            candidates.pop(worst)
            ballots = np.delete(ballots, worst, 1)
        return candidates[0]

    def compute_ballots_honest(self, utilities, poll=None):
        return np.flip(utilities.argsort(), axis=1).argsort()

class Approval(VotingMethod):

    def __init__(self, behaviour):
        super().__init__(behaviour)

    def compute_winner(self, ballots):
        return ballots.sum(axis=0).argmax()

    def compute_ballots_honest(self, utilities, poll=None):
        tolerance = (utilities.max(axis=1) + utilities.min(axis=1)) / 2
        return (utilities > tolerance[:, None]).astype(int)

class Score(VotingMethod):

    def __init__(self, behaviour, max_score=5):
        super().__init__(behaviour)
        self.max_score = max_score

    def compute_winner(self, ballots):
        return ballots.sum(axis=0).argmax()

    def compute_ballots_honest(self, utilities, poll=None):
        ballots = utilities - utilities.min(axis=1)[:, None]
        ballots = ballots / ballots.max(axis=1)[:, None]
        ballots = (ballots*(self.max_score + 1)).astype(int)
        ballots[ballots == (self.max_score + 1)] -= 1
        return ballots

    def compute_ballots_strategy(self, utilities, poll=None):
        front_runners = np.argsort(poll)[-2:]
        idx = front_runners[utilities[:, front_runners].argmax(axis=1)]
        utilities[range(len(utilities)), idx] = 1
        idx = front_runners[utilities[:, front_runners].argmin(axis=1)]
        utilities[range(len(utilities)), idx] = 0
        return self.compute_ballots_honest(utilities)

    def __repr__(self):
        return f'{self.__class__.__name__}{self.max_score}'


class STAR(VotingMethod):

    def __init__(self, behaviour, max_score=5):
        super().__init__(behaviour)
        self.max_score = max_score

    def compute_winner(self, ballots):
        return ballots.sum(axis=0).argmax()

    def compute_ballots_honest(self, utilities, poll=None):
        ballots = utilities - utilities.min(axis=1)[:, None]
        ballots = ballots / ballots.max(axis=1)[:, None]
        ballots = (ballots*(self.max_score + 1)).astype(int)
        ballots[ballots == (self.max_score + 1)] -= 1
        return ballots

    def compute_ballots_strategy(self, utilities, poll=None):
        front_runners = np.argsort(poll)[-2:]
        idx = front_runners[utilities[:, front_runners].argmax(axis=1)]
        utilities[range(len(utilities)), idx] = 1
        idx = front_runners[utilities[:, front_runners].argmin(axis=1)]
        utilities[range(len(utilities)), idx] = 0
        return self.compute_ballots_honest(utilities)

    def __repr__(self):
        return f'{self.__class__.__name__}{self.max_score}'
