from scipy.linalg import logm
import numpy as np

from sklearn.base import TransformerMixin
from pyriemann.tangentspace import TangentSpace


class Riemann(TransformerMixin):
    def __init__(self, n_fb=9, metric='riemann'):
        self.n_fb = n_fb
        self.ts = [TangentSpace(metric=metric) for fb in range(n_fb)]

    def fit(self, X, y):
        for fb in range(self.n_fb):
            self.ts[fb].fit(X[:, fb, :, :])
        return self

    def transform(self, X):
        n_sub, n_fb, p, _ = X.shape
        Xout = np.empty((n_sub, n_fb, p*(p+1)//2))
        for fb in range(n_fb):
            Xout[:, fb, :] = self.ts[fb].transform(X[:, fb, :, :])
        return Xout.reshape(n_sub, -1)  # (sub, fb * c*(c+1)/2)

    def inverse_transform(self, X):
        n_sub, nfeatures = X.shape
        n_ts = int(nfeatures / self.n_fb)
        n_channels = int(-.5 + np.sqrt(.25 + 2*n_ts)) # c*(c+1)/2 = nts

        X = X.reshape(n_sub, self.n_fb, n_ts)

        Xout = np.empty((n_sub, self.n_fb, n_channels, n_channels))
        for fb in range(self.n_fb):
            Xout[:, fb, :, :] = self.ts[fb].inverse_transform(X[:, fb, :])
        return Xout


class Diag(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        Xout = np.empty((n_sub, n_fb, n_compo))
        for sub in range(n_sub):
            for fb in range(n_fb):
                Xout[sub, fb] = np.diag(X[sub, fb])
        return Xout.reshape(n_sub, -1)  # (sub, fb * n_compo)


class LogDiag(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        Xout = np.empty((n_sub, n_fb, n_compo))
        for sub in range(n_sub):
            for fb in range(n_fb):
                Xout[sub, fb] = np.log10(np.diag(X[sub, fb]))
        return Xout.reshape(n_sub, -1)  # (sub, fb * n_compo)


class NaiveVec(TransformerMixin):
    def __init__(self, method, n_fb=9):
        self.method = method
        self.n_fb = n_fb
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, n_compo, _ = X.shape
        q = int(n_compo * (n_compo+1) / 2)
        Xout = np.empty((n_sub, n_fb, q))
        for sub in range(n_sub):
            for fb in range(n_fb):
                if self.method == 'upper':
                    Xout[sub, fb] = X[sub, fb][np.triu_indices(n_compo)]
                elif self.method == 'upperlog':
                    logmat = logm(X[sub, fb])
                    Xout[sub, fb] = logmat[np.triu_indices(n_compo)]
                elif self.method == 'logdiag+upper':
                    logdiag = np.log10(np.diag(X[sub, fb]))
                    upper = X[sub, fb][np.triu_indices(n_compo, k=1)]
                    Xout[sub, fb] = np.concatenate((logdiag, upper), axis=None)
        return Xout.reshape(n_sub, -1)  # (sub, fb * c*(c+1)/2)

    def inverse_transform(self, X):
        n_sub, nfeatures = X.shape
        n_ts = int(nfeatures / self.n_fb)
        n_channels = int(-.5 + np.sqrt(.25 + 2*n_ts)) # c*(c+1)/2 = nts

        X = X.reshape(n_sub, self.n_fb, n_ts)

        Xout = np.zeros((n_sub, self.n_fb, n_channels, n_channels))

        for sub in range(n_sub):
            for fb in range(self.n_fb):
                if self.method == 'upper':
                    Xout[sub, fb][np.triu_indices(n_channels)] = X[sub, fb] * 0.5 # share weights equally 
                    Xout[sub, fb] += Xout[sub, fb].T
                else:
                    raise NotImplementedError()
        return Xout        
