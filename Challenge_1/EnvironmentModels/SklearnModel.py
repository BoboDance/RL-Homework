from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel


class SklearnModel(object):

    def __init__(self, type="rf"):

        if type == "gp":
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            self.model = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b")
        elif type == "rf":
            self.model = RandomForestRegressor(n_estimators=100)
        else:
            raise NotImplementedError("Model not known.")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
