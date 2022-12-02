from abc import ABC, abstractmethod

import numpy as np


class VotingMethod(ABC):

    def __init__(self, behaviour, p=1):
        behaviour_method = f'compute_ballots_{behaviour}'
        assert behaviour_method in dir(self)
        assert 0 < p <= 1

        self.behaviour = behaviour
        self.compute_ballots = getattr(self, behaviour_method)
        self.p = p

    @abstractmethod
    def compute_winner(self, ballots):
        pass

    @abstractmethod
    def compute_ballots_honest(self, util, poll=None):
        pass

    def split_util(func):
        def wrapper(self, util, poll):
            if self.p == 1:
                ballots = func(self, util, poll)
            else:
                i = int(self.p*util.shape[0])
                strat_util, honest_util = util[:i], util[i:]
                honest_ballots = self.compute_ballots_honest(honest_util)
                strat_ballots = func(self, strat_util, poll)
                ballots = np.append(strat_ballots, honest_ballots)
            return ballots
        return wrapper

    def __repr__(self):
        return f'{self.__class__.__name__} {self.behaviour} {int(100*self.p)}%'


class PluralityVotingMethod(VotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_ballots_honest(self, util, poll=None):
        return util.argmax(axis=1)

    @VotingMethod.split_util
    def compute_ballots_strategic(self, util, poll):
        fr = poll.argsort()[-2:]
        return fr[util[:, fr].argmax(axis=1)]

    @VotingMethod.split_util
    def compute_ballots_strategic1s(self, util, poll):
        fr = poll.argsort()[-2:]
        strat_ballots = fr[util[:, fr].argmax(axis=1)]
        idx = strat_ballots != fr[0]
        strat_ballots[idx] = self.compute_ballots_honest(util[idx])
        return strat_ballots

class OrdinalVotingMethod(VotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_ballots_honest(self, util, poll=None):
        return util.argmax(axis=1)

    def compute_ballots_strategic(self, util, poll):
        pass

    def compute_ballots_strategic_1s(self, util, poll):
        pass

class CardinalVotingMethod(VotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_ballots_honest(self, util, poll=None):
        return util.argmax(axis=1)

    def compute_ballots_strategic(self, util, poll):
        pass

    def compute_ballots_strategic_1s(self, util, poll):
        pass


class FPTP(PluralityVotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        return np.bincount(ballots).argmax()

class Borda(OrdinalVotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        pass

class IRV(OrdinalVotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        pass

class Score(CardinalVotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        pass

class STAR(CardinalVotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        pass
