from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel


class GPModel(object):

    def __init__(self):
        kernel = ConstantKernel() + RBF() + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b")

    def fit(self, X, y):
        self.gp.fit(X, y)

    def predict(self, X):
        return self.gp.predict(X)

    def discretize(self, x):
        raise NotImplementedError
