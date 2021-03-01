# %%

from os import makedirs
import os.path as op
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# project imports
from library.simuls import generate_covariances, generate_covariances_and_a
from library.spfiltering import ProjIdentitySpace, ProjSPoCSpace
from library.featuring import Diag, LogDiag, Riemann
from library.pattern import PatternScorer
import config as cfg

# %% parameters
sampling_freq = 1 # Hz
n_matrices = 100  # Number of matrices
f_powers = 'log'  # link function between the y and the source powers
rng = 4
sigma = 0
n_channels = 5
n_sources = 1
distance_A_id = 1.0

scoring = 'neg_mean_absolute_error'

out_path = op.join(cfg.path_outputs, 'simuls', 'patternnoise')

if not op.exists(out_path):
    makedirs(out_path)


# %% pipelines

# define spatial filters
identity = ProjIdentitySpace()
spoc = ProjSPoCSpace(n_compo=n_channels, scale='auto', reg=0, shrink=0)

n_compo = n_channels

# define featuring
diag = Diag()
logdiag = LogDiag()
riemann = Riemann(n_fb=1, metric='riemann')

sc = StandardScaler()

# define algo
dummy = DummyRegressor()
ridge = RidgeCV(alphas=np.logspace(-5, 3, 25), scoring=scoring)

# define models
pipelines = {
    'dummy': make_pipeline(identity, logdiag, sc, dummy),
    'diag': make_pipeline(identity, logdiag, sc, ridge),
    'spoc': make_pipeline(spoc, logdiag, sc, ridge),
    'riemann': make_pipeline(identity, riemann, sc, ridge)
}                        

# %% cross-validation simulations for sigma parameter

noises_A = np.logspace(-3, 0, 10)

# Run experiments
n_cv = 10
n_methods = len(pipelines)

resdf = pd.DataFrame(index=range(n_methods * len(noises_A)), \
    columns = ['method', 'nonlinearity', 'sigma', 'noise_A', 'n_sources', 'n_compo', \
        'target_score_mu', 'target_score_sd', 'pattern_score_mu', 'pattern_score_sd'])
resix = 0

for j, noise_A in enumerate(noises_A):    

    X, y, A = generate_covariances_and_a(n_matrices, n_channels, n_sources,
                                sigma=sigma, distance_A_id=distance_A_id,
                                f_p=f_powers, direction_A=None,
                                noise_A=noise_A, rng=rng)


    X = X[:, None, :, :]
    for i, (name, pipeline) in enumerate(pipelines.items()):
        print('noise_A = {}, {} method'.format(noise_A, name))

        a = A[:,0:n_sources,None] / np.linalg.norm(A[:,0:n_sources,None], axis=0)

        pttrn_scorer = PatternScorer(a=a, name=name)

        scoring = {'mae': 'neg_mean_absolute_error', 
                   'pattrn': pttrn_scorer }

        sc = cross_validate(pipeline, X, y, scoring=scoring,
                             cv=n_cv, n_jobs=-1, error_score=np.nan, return_train_score=True)                             
        
        resdf.loc[resix,('method', 'nonlinearity', 'sigma', 'noise_A', 'n_sources', 'n_compo')] = \
            (name, f_powers, sigma, noise_A, n_sources, n_compo)
        resdf.loc[resix,('target_score_mu', 'target_score_sd')] = (- np.mean(sc['test_mae']), np.std(sc['test_mae']))
        resdf.loc[resix,('pattern_score_mu', 'pattern_score_sd')] = (np.mean(1. - sc['train_pattrn']), np.std(1. - sc['train_pattrn']))

        

        resix += 1

# %% save results
resdf.to_csv(op.join(out_path, 'scores.csv'), index=False) 
 
