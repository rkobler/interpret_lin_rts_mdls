# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import mne
from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path

from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold, cross_validate
from sklearn.metrics import mean_absolute_error

# project imports
from library.spfiltering import ProjIdentitySpace, ProjSPoCSpace, ProjCommonSpace
from library.featuring import Diag, LogDiag, Riemann
from library.pattern import PatternScorer
import config as cfg

# %% load the dataset
fname = data_path() + '/SubjectCMC.ds'

out_path = os.path.join(cfg.path_outputs, 'ds-cmc')
if not os.path.exists(out_path):
    os.makedirs(out_path)

raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.).load_data()  # crop for memory purposes

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGlft'])
emg.filter(20., None, fir_design='firwin')

# Filter MEG data to focus on beta band
raw.pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
raw.filter(15., 30., fir_design='firwin')

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=8)
emg_epochs = Epochs(emg, events, tmin=0., tmax=1.500, baseline=None)

# Prepare classification
X = meg_epochs.get_data() * 1e12
y = emg_epochs.get_data().var(axis=2)[:, 0] * 1e12 # target is EMG power

# %% featuring

n_channels = X.shape[1]
n_epochs = X.shape[0]
winlen = X.shape[2]

# compute covariance matrices
C =  X @ X.swapaxes(-1, -2) / n_channels

# perform OAS shrinkage
mu = np.trace(C, axis1=-1, axis2=-2) / n_channels
alpha = np.square(C).mean(axis=(-1, -2))
num = alpha + np.square(mu)
den = (winlen + 1.) * (alpha - np.square(mu) / n_channels)
shrinkage = (num / den).clip(0, 1)

# take the average shrinkage coefficients over all covariance matrices
sf = np.mean(shrinkage)
st = np.mean(mu)

C = (1. - sf) * C.copy()
C += sf * (np.eye(n_channels) * st)[None,:,:]

# only 1 frequency band
C = C[:,None,:]

# %% define the pipelines

scoring = 'neg_mean_absolute_error'

# define spatial filters
common = ProjCommonSpace(n_compo=48, scale='auto', reg=1e-5)
identity = ProjIdentitySpace()
spoc = ProjSPoCSpace(n_compo=4, scale='auto', reg=1e-5, shrink=0.5)

# define featuring
diag = Diag()
logdiag = LogDiag()
riemann = Riemann(n_fb=1, metric='riemann')

sc = StandardScaler()

# define algo
logreg = RidgeCV(alphas=np.logspace(-5, 3, 25), scoring=scoring)

# define models
pipelines = {
    'diag': make_pipeline(identity, logdiag, sc, logreg),
    'spoc': make_pipeline(spoc, logdiag, sc, logreg),
    'riemann': make_pipeline(common, riemann, sc, logreg)
}                        

# %% decoding experiments

n_cv = 10
n_compos = range(2,64+1,2)

index = range(n_cv * len(pipelines) * len(n_compos))

resdf = pd.DataFrame(columns=['method', 'components', 'fold', 'r2', 'mae'], index = index)
resdf = resdf.astype({'r2': 'float64', 'mae': 'float64'})

resix = 0
for i, (name, pipeline) in enumerate(pipelines.items()):

    for n_compo in n_compos:

        print('{} method {} components'.format(name, n_compo))

        pipeline.steps[0][1].n_compo = n_compo

        scoring = {'mae': 'neg_mean_absolute_error', 
                   'r2': 'r2' }
        sc = cross_validate(pipeline, C, y, scoring=scoring,
                            cv=n_cv, n_jobs=-1, error_score=np.nan)  

        idxs = range(resix, resix+n_cv)

        resdf.loc[idxs,('method', 'components')] = \
            (name, n_compo)
        resdf.loc[idxs,'r2'] = sc['test_r2']
        resdf.loc[idxs,'mae'] = - sc['test_mae']
        resdf.loc[idxs,'fold'] = range(n_cv)

        resix += n_cv

resdf.to_csv(os.path.join(out_path, 'scores.csv'), index=False) 

# %% pattern experiments

# resdf = pd.read_csv(os.path.join(out_path, 'scores.csv')) 

methods = ['spoc', 'riemann']

pattern_res = []

avgresdf = resdf.groupby(['method', 'components']).agg({'r2' : 'mean'}).reset_index()

for method in methods:

    pipeline = pipelines[method]

    # find the model with the highest average score and compute the patterns
    n_compo_max = avgresdf.loc[avgresdf.loc[avgresdf.method == method,'r2'].idxmax()].components.astype(np.int64)

    pipeline.steps[0][1].n_compo = n_compo_max
    pipeline.fit(C,y)

    # compute the patterns
    (a_hat, evals) = PatternScorer(None, method).compute_patterns(pipeline, C, scaled=False, return_evals=True)

    # store the results for plotting
    y_hat = cross_val_predict(pipeline, C, y, cv=n_cv, n_jobs=-1)
    n = min(n_compo_max, 8)

    pattern_res += [(a_hat[:,:,0],n, f'{method} {n_compo_max}', np.abs(evals), y_hat)]


# %% pattern plotting

info = meg_epochs.info.copy()
info['sfreq'] = 1

ncols = 4

for (a_hat, n_comp, title, evals, y_hat) in pattern_res:

    nrows = np.ceil(n_comp/ncols).astype(np.int64) + 1

    fig = plt.figure(tight_layout=True, figsize=[10, 2*nrows])
    gs = fig.add_gridspec(nrows, ncols)

    ax = fig.add_subplot(gs[0,:-1])
    times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
    ax.plot(times, y_hat, label='decoded')
    ax.plot(times, y, label='target')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('EMG power')
    ax.set_title(title)
    plt.legend()   

    if evals is not None:
        ax = fig.add_subplot(gs[0,-1])
        ax.plot(evals, '.-')
        ax.set_xlabel('component')
        ax.set_title('eigenvalues')

    for col in range(n_comp):
        ax = fig.add_subplot(gs[1+col//ncols, col %  ncols])
        mne.viz.plot_topomap(a_hat[:,col], info, show = False, axes = ax, sensors=False, extrapolate='local')
        ax.set_xlabel(f'component {col}')

    fig.savefig(os.path.join(out_path, 'fig_' + title +'.pdf'))
