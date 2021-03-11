import numpy as np
from sklearn.metrics import make_scorer
from scipy.linalg import pinv, eigh, svd
from sklearn.covariance import OAS, EmpiricalCovariance

from library.spfiltering import ProjCommonSpace, ProjIdentitySpace, ProjSPoCSpace, ProjCSPSpace, shrink

class PatternScorer:
    def __init__(self, a, name, return_mean = True):
        self.a = a
        self.name = name
        self.return_mean = return_mean

    def __call__(self, pipeline, X, y_true, sample_weight=None):

        n_sources = self.a.shape[1]
        n_fb = X.shape[1]

        if type(pipeline.steps[0][1]) is ProjCommonSpace:
            n = min(pipeline.steps[0][1].n_compo, n_sources)
        if type(pipeline.steps[0][1]) is ProjCSPSpace:
            n = min(pipeline.steps[0][1].n_compo, n_sources)
        if type(pipeline.steps[0][1]) is ProjSPoCSpace:
            n = min(pipeline.steps[0][1].n_compo, n_sources)                        
        elif type(pipeline.steps[0][1]) is ProjIdentitySpace:
            n = n_sources

        a_hat = self.compute_patterns(pipeline, X)

        if a_hat is None:
            return np.array(1.) * np.nan            

        scores = np.zeros((n_sources))

        for f in range(n_fb):

            # check if the a_hats span the subspaces of the a's
            # project the true a's on the a_hats
            # since all are normalized, the dot products is in the range [-1,1]
            # by taking the maximal abs in-product, we can see if each a has a matching a_hat
            scores += 1./n_fb * np.max(np.abs(self.a[:,:,f].T @ a_hat[:,0:n,f]), axis = 1)

        if self.return_mean:
            return np.mean(scores)
        else:
            return scores

    def compute_patterns(self, pipeline, X, scaled = False, return_evals = False):

        n_fb = X.shape[1]

        if self.name == 'riemann':

            # extract model parameters
            rs_betas = np.squeeze(pipeline.steps[-1][1].coef_)[None,:]

            # project the features to the last layer 
            # before training the regression model
            Xproj = X.copy()
            for _, step in pipeline.steps[:-1]:
                Xproj = step.transform(Xproj)

            # estimate the covariance of the latent features
            covest = OAS().fit(Xproj)
            Cxx = covest.covariance_
            # convert the regression coefficients to patterns
            rs_pttrn = Cxx @ rs_betas.T / (rs_betas @ Cxx @ rs_betas.T)

            rs_origin = np.zeros(rs_betas.shape)
            rs_points = np.concatenate((rs_origin, rs_pttrn.T), axis=0)

            # regression space 2 tangent space
            ts_points = pipeline.steps[-2][1].inverse_transform(rs_points)
            # tangent space 2 SPD space
            spd_points = pipeline.steps[-3][1].inverse_transform(ts_points)

            n_compo = spd_points.shape[2]

            a_hat = np.zeros((X.shape[2], n_compo, n_fb))
            eigvals = np.zeros((n_compo, n_fb))

            # compute linear weights and the patterns
            # for each frequency band
            for f in range(n_fb):
                evals, evecs = eigh(spd_points[1,f], spd_points[0,f])

                # search for largest eigenvale or inverse eigenvalue
                evals = np.max(np.concatenate((evals[:,None], 1/evals[:,None]), axis = 1), axis = 1)

                ix = np.argsort(np.abs(evals))[::-1]
                evecs = evecs[:, ix]
                evals = evals[ix]

                # pick largest or smalled eigenvectors (both are possible solutions)
                if scaled:
                    pttrns = pinv(evecs).T @ np.sqrt(np.diag(evals))
                else:
                    pttrns = pinv(evecs).T

                if type(pipeline.steps[0][1]) is ProjCommonSpace:
                    # invert common space projection
                    # filters = pipeline.steps[0][1].filters_[0]
                    # inv_filters = pinv(filters)
                    inv_filters = pipeline.steps[0][1].patterns_[f]
                    pttrns = inv_filters.T @ pttrns            

                a_hat[:,:,f] = pttrns / np.linalg.norm(pttrns, 2, 0)
                eigvals[:,f] = evals

        elif self.name == 'spoc' or self.name == 'csp':

            n_compo = pipeline.steps[0][1].n_compo

            a_hat = np.zeros((X.shape[2], n_compo, n_fb))
            eigvals = np.zeros((n_compo, n_fb))

            for f in range(n_fb):

                a_hat[:,:,f] = pipeline.steps[0][1].patterns_[f].T
                a_hat[:,:,f] /= np.linalg.norm(a_hat[:,:,f], 2, 0)
                eigvals[:,f] = pipeline.steps[0][1].evals_[f]

        else:
            return None

        if return_evals:
            return (a_hat, eigvals)
        else:
            return a_hat