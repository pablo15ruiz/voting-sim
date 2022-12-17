from abc import ABC, abstractmethod

import numpy as np


class VotingMethod(ABC):

    def __init__(self, behaviour, p):
        self.name = self.__class__.__name__
        self.behaviour = behaviour
        self.p = p

        if f'compute_ballots_{behaviour}' not in dir(self):
            raise NameError(f'Method {self} not implemented')
        if behaviour == 'honest' and p != 1:
            raise ValueError('`p` must be 1 for honest methods')
        if not 0 < p <= 1:
            raise ValueError('`p` must be beetween 0 and 1')

    @abstractmethod
    def compute_winner(self, ballots):
        pass

    @abstractmethod
    def compute_ballots_honest(self, util, poll):
        pass

    def compute_ballots(self, util, poll):
        if self.behaviour == 'honest':
            ballots = self.compute_ballots_honest(util)
        else:
            method = f'compute_ballots_{self.behaviour}'
            i = int(self.p*util.shape[0])
            ballots = self.compute_ballots_honest(util[i:])
            behaviour_ballots = getattr(self, method)(util[:i], poll)
            ballots = np.append(behaviour_ballots, ballots, axis=0)

        return ballots

    def __str__(self):
        m = f'{self.name} {self.behaviour}'
        if self.behaviour != 'honest':
            m += f' {int(100*self.p)}%'
        return m


class PluralityVotingMethod(VotingMethod):

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_ballots_honest(self, util, poll=None):
        return util.argmax(axis=1)

    def compute_ballots_strategic(self, util, poll):
        fr = poll[-2:]
        return fr[util[:, fr].argmax(axis=1)]

    def compute_ballots_strategic1s(self, util, poll):
        ballots = self.compute_ballots_strategic(util, poll)
        fr = poll[-2:]
        idx = ballots == fr[1]
        ballots[idx] = self.compute_ballots_honest(util[idx])
        return ballots

class OrdinalVotingMethod(VotingMethod):
    '''
    Convert `util` to `ballots`.
    Return ranked ballot (first/worst to last/best) by arg.
    '''

    def __init__(self, behaviour, p):
        super().__init__(behaviour, p)

    def compute_ballots_honest(self, util, poll=None):
        return util.argsort()

    def compute_ballots_strategic(self, util, poll):
        n_vot, n_cand = util.shape
        fr = poll[-2:]
        ballots = self.compute_ballots_honest(util)
        mask = np.isin(ballots, fr, invert=True)
        ballots = ballots[mask].reshape((n_vot, n_cand-2))
        pref = fr[util[:, fr].argmax(axis=1)]
        temp = pref.copy()
        temp[pref == fr[0]] = fr[1]
        temp[pref == fr[1]] = fr[0]
        ballots = np.c_[temp, ballots, pref]
        return ballots

    def compute_ballots_strategic1s(self, util, poll):
        ballots = self.compute_ballots_strategic(util, poll)
        fr = poll[-2:]
        idx = ballots[:, -1] == fr[1]
        ballots[idx] = self.compute_ballots_honest(util[idx])
        return ballots

class CardinalVotingMethod(VotingMethod):

    def __init__(self, behaviour, p, max_score):
        super().__init__(behaviour, p)
        self.max_score = max_score
        self.name += f'({max_score})'

    def compute_ballots_honest(self, util, poll=None):
        ballots = util - util.min(axis=1)[:, None]
        ballots = ballots/ballots.max(axis=1)[:, None]
        ballots = (ballots*(self.max_score + 1)).astype(int)
        ballots[ballots == (self.max_score + 1)] -= 1
        return ballots

    def compute_ballots_strategic(self, util, poll):
        fr = poll[-2:]
        pref = fr[util[:, fr].argmax(axis=1)]
        temp = [fr[0] if x == fr[1] else fr[1] for x in pref]
        ballots = self.compute_ballots_honest(util)
        n_vot = util.shape[0]
        ballots[range(n_vot), pref] = self.max_score
        ballots[range(n_vot), temp] = 0
        return ballots

    def compute_ballots_strategic1s(self, util, poll):
        ballots = self.compute_ballots_strategic(util, poll)
        fr = poll[-2:]
        pref = fr[util[:, fr].argmax(axis=1)]
        idx = pref == fr[1]
        ballots[idx] = self.compute_ballots_honest(util[idx])
        return ballots

class FPTP(PluralityVotingMethod):

    def __init__(self, behaviour, p=1):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        return np.bincount(ballots).argmax()

class Borda(OrdinalVotingMethod):

    def __init__(self, behaviour, p=1):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        return np.flip(ballots, axis=1).argsort().sum(axis=0).argmin()

class IRV(OrdinalVotingMethod):

    def __init__(self, behaviour, p=1):
        super().__init__(behaviour, p)

    def compute_winner(self, ballots):
        n_vot, n_cand = ballots.shape
        cand = list(range(n_cand))
        n_cand_temp = n_cand
        while n_cand_temp > 1:
            count = np.bincount(ballots[:, -1], minlength=n_cand)
            if count.max() > n_vot/2:
                winner = count.argmax()
                break
            worst = cand[count[cand].argmin()]
            cand.remove(worst)
            n_cand_temp -= 1
            ballots = ballots[ballots != worst].reshape((n_vot, n_cand_temp))
            if n_cand_temp == 1:
                winner = cand[0]

        return winner

class Score(CardinalVotingMethod):

    def __init__(self, behaviour, p=1, max_score=5):
        super().__init__(behaviour, p, max_score)
        if max_score == 1:
            self.name = 'Approval'

    def compute_winner(self, ballots):
        return ballots.sum(axis=0).argmax()

class STAR(CardinalVotingMethod):

    def __init__(self, behaviour, p=1, max_score=5):
        super().__init__(behaviour, p, max_score)

    def compute_winner(self, ballots):
        fr = ballots.sum(axis=0).argsort()[-2:]
        ballots = ballots[:, fr]
        ballots = ballots[ballots[:, 0] != ballots[:, 1]]
        if not len(ballots):
            winner = fr[0]
        else:
            winner = fr[np.bincount(ballots.argmax(axis=1)).argmax()]

        return winner

class ThreeTwoOne(CardinalVotingMethod):

    def __init__(self, behaviour, p=1, max_score=2):
        assert max_score == 2
        super().__init__(behaviour, p, max_score)
        self.name = '3-2-1'

    def compute_winner(self, ballots):
        semi = (ballots == self.max_score).sum(axis=0).argsort()[-3:]
        final = semi[(ballots[:, semi] == 0).sum(axis=0).argsort()[:2]]
        ballots = ballots[:, final]
        ballots = ballots[ballots[:, 0] != ballots[:, 1]]
        if not len(ballots):
            winner = final[0]
        else:
            winner = final[np.bincount(ballots.argmax(axis=1)).argmax()]
        return winner

class MajorityJudgment(CardinalVotingMethod):

    def __init__(self, behaviour, p=1, max_score=5):
        super().__init__(behaviour, p, max_score)

    def compute_winner(self, ballots):
        ballots = np.sort(ballots, axis=0)
        tie = np.arange(ballots.shape[1])
        for idx in reversed(range(int(len(ballots)/2))):
            alpha = ballots[idx, tie]
            tie = tie[np.flatnonzero(alpha == alpha.max())]
            if len(tie) == 1:
                break

        return tie[0]

class SmithScore(CardinalVotingMethod):

    def __init__(self, behaviour, p=1, max_score=5):
        super().__init__(behaviour, p, max_score)

    def compute_winner(self, ballots):
        smith = (ballots[:, None] < ballots[:, :, None]).sum(axis=0)
        smith = ((smith - smith.T) >= 0).sum(axis=1)
        smith = np.flatnonzero(smith == smith.max())
        return smith[ballots[:, smith].sum(axis=0).argmax()]
