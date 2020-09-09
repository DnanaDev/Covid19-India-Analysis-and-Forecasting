"""
All the machine learning, statistical models for the Covid Forecasting Dashboard.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import numpy as np

# curve fit function from scipy.optimize to fit a function using nonlinear least squares.
from scipy.optimize import curve_fit


class SigmoidCurveFit(BaseEstimator, RegressorMixin):
    """Sklearn Wrapper to fit a General Logistic curve of the form $f{(x)} = \frac{L}{1 + e^{-k(x - x_0)}}$.
    Where x_0 = the value of the sigmoids' midpoint.
          L = the curve's maximum value.
          k = the logistic growth rate or steepness of the curve.
    The parameters aer returned in the popt variable after fitting the sigmoid.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        # to check if model has been fit, will not be none after fit
        self.popt = None

    def sigmoid(self, x, L, x0, k):
        return L / (1 + np.exp(-k * (x - x0)))

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # need to be flattened again as scipy curve fit takes flat 1-D array
        self.X_ = np.ravel(X)
        self.y_ = y

        # fit parameters and covariance matrix
        self.popt, self.pcov = curve_fit(self.sigmoid, self.X_, self.y_)

    def predict(self, X):
        # Check is fit had been called
        if self.popt is None:
            check_is_fitted(self, 'model_')
        return self.sigmoid(X, *self.popt)

    def get_sigmoid_params(self):
        # Check is fit had been called
        if self.popt is None:
            check_is_fitted(self, 'model_')
        return {'L': self.popt[0], 'x0': self.popt[1], 'k': self.popt[2]}