import copy as cp
import numpy as np
from scipy.linalg import pinv, eigh

from sklearn.base import TransformerMixin


def shrink(cov, alpha):
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.trace(cov) * np.eye(n) / n
    return shrink_cov


def fstd(y):
    y = y.astype(np.float32)
    y -= y.mean(axis=0)
    y /= y.std(axis=0)
    return y


def _get_scale(X, scale):
    if scale == 'auto':
        scale = 1 / np.mean([[np.trace(y) for y in x] for x in X])
    return scale


class ProjIdentitySpace(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


class ProjCommonSpace(TransformerMixin):
    def __init__(self, scale=1, n_compo=71, reg=1e-7):
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y):
        _, n_fb, _, _ = X.shape
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C = covsfb.mean(axis=0)
            eigvals, eigvecs = eigh(C)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs /= np.linalg.norm(evecs, axis=0)[None, :]

            evecs = evecs[:, :self.n_compo]
            pattern = C @ evecs @ np.linalg.inv(evecs.T @ C @ evecs)

            self.filters_.append(evecs.T)  # (fb, compo, chan) row vec
            self.patterns_.append(pattern.T)  # (fb, compo, chan)

        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ Xs[sub, fb] @ filters.T
                Xout[sub, fb] += self.reg * np.eye(self.n_compo)
        return Xout  # (sub , fb, compo, compo)


class ProjSPoCSpace(TransformerMixin):
    def __init__(self, shrink=0, scale=1, n_compo=71, reg=1e-7):
        self.shrink = shrink
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y):
        n_sub, n_fb, _, _ = X.shape
        target = fstd(y)
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        self.evals_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C = covsfb.mean(axis=0)
            Cz = np.mean(covsfb * target[:, None, None], axis=0)
            C = shrink(C, self.shrink)
            eigvals, eigvecs = eigh(Cz, C)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            eigvals = eigvals[ix]
            pattern = pinv(evecs.T).T
            evecs = evecs[:, :self.n_compo].T
            
            self.filters_.append(evecs)  # (fb, compo, chan) row vec
            self.patterns_.append(pattern[:self.n_compo,:])  # (fb, compo, chan)
            self.evals_.append(eigvals[:self.n_compo])
        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ Xs[sub, fb] @ filters.T
                Xout[sub, fb] += self.reg * np.eye(self.n_compo)
        return Xout  # (sub, fb, compo, compo)


class ProjCSPSpace(TransformerMixin):
    def __init__(self, shrink=0, scale=1, n_compo=71, reg=1e-7):
        self.shrink = shrink
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y):
        _, n_fb, _, _ = X.shape
        
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        self.evals_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C0 = covsfb[y == 0].mean(axis=0)
            C1 = covsfb[y != 0].mean(axis=0)
            C0 = shrink(C0, self.shrink)
            C1 = shrink(C1, self.shrink)
            eigvals, eigvecs = eigh(C1, C0)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            eigvals = np.max(np.concatenate((eigvals[:,None], 1/eigvals[:,None]), axis = 1), axis = 1)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            eigvals = eigvals[ix]
            pattern = pinv(evecs.T).T
            evecs = evecs[:, :self.n_compo].T
            
            self.filters_.append(evecs)  # (fb, compo, chan) row vec
            self.patterns_.append(pattern[:self.n_compo,:])  # (fb, compo, chan)
            self.evals_.append(eigvals[:self.n_compo])

        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ Xs[sub, fb] @ filters.T
                Xout[sub, fb] += self.reg * np.eye(self.n_compo)
        return Xout  # (sub, fb, compo, compo)  

