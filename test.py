from simulation import *
from voter_models import *
from voting_methods import *

import numpy as np

np.random.seed(1)

n_vot = 100
n_cand = 4
n_iter = 10000

methods = [
    FPTP('honest'),
    FPTP('strategy'),
    Borda('honest'),
    Condorcet('honest'),
    IRV('honest'),
    Approval('honest'),
    Score('honest', max_score=5),
    Score('strategy', max_score=5),
    Score('honest', max_score=50),
    Score('honest', max_score=3),
]
voter_model = ImpartialCulture()
sim = Simulation(n_vot, n_cand, n_iter, methods, voter_model)
sim.simulate()
sim.compute_metrics()
sim.show_results()

# method = Score('strategy', max_score=5)
# utilities = ImpartialCulture().generate_utilities(n_vot, n_cand)
# elec = Election(n_vot, n_cand, utilities, poll=True)
# winner = elec.compute_winner(method)
# print(winner)
