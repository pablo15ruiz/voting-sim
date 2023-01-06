from simulation import *
from voter_models import *
from voting_methods import *

import numpy as np

methods = [
    FPTP('honest'),
    Borda('strategic'),
    IRV('strategic', p=0.5),
    Score('strategic1s', max_score=1),
    Score('honest'),
    STAR('honest'),
    ThreeTwoOne('strategic'),
    MajorityJudgment('honest'),
    SmithScore('strategic1s', p=0.3)
]

np.random.seed(1)

n_vot = 300
n_cand = 6
n_iter = 300
voter_model = ClusteredSpatialModel(n_dim=4)

sim = Simulation(n_vot, n_cand, n_iter, methods, voter_model)
sim.simulate()
sim.results()
