from simulation import *
from voter_models import *
from voting_methods import *

import numpy as np
import profile

np.random.seed(1)

n_vot = 300
n_cand = 6
n_iter = 300

voter_model = ClusteredSpatialModel()
voter_model = ImpartialCulture()
methods = [
    FPTP('honest'),
    FPTP('strategic'),
    FPTP('strategic', p=0.5),
    FPTP('strategic1s'),
    FPTP('strategic1s', p=0.5),
    Borda('honest'),
    Borda('strategic'),
    Borda('strategic', p=0.5),
    Borda('strategic1s'),
    Borda('strategic1s', p=0.5),
    IRV('honest'),
    IRV('strategic'),
    IRV('strategic', p=0.5),
    IRV('strategic1s'),
    IRV('strategic1s', p=0.5),
    Score('honest', max_score=1),
    Score('strategic', max_score=1),
    Score('strategic', max_score=1, p=0.5),
    Score('strategic1s', max_score=1),
    Score('strategic1s', max_score=1, p=0.5),
    Score('honest'),
    Score('strategic'),
    Score('strategic', p=0.5),
    Score('strategic1s'),
    Score('strategic1s', p=0.5),
    STAR('honest'),
    STAR('strategic'),
    STAR('strategic', p=0.5),
    STAR('strategic1s'),
    STAR('strategic1s', p=0.5)
]

sim = Simulation(n_vot, n_cand, n_iter, methods, voter_model)
sim.simulate()
sim.results()
# sim.to_csv()
