"""
All the machine learning, statistical models for the Covid Forecasting Dashboard.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# curve fit function from scipy.optimize to fit a function using nonlinear least squares.
from scipy.optimize import curve_fit

""" Sigmoid Fitting Functions
"""


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


""" Growth Factor Prediction Functions 
"""


def growth_factor_features(gf_df):
    """All features should be available at inference time"""
    # creating lagged features
    gf_df['Lag_1days'] = gf_df.shift(1)
    gf_df['Lag_2days'] = gf_df['Growth_Factor'].shift(2)
    gf_df['Lag_3days'] = gf_df['Growth_Factor'].shift(3)
    gf_df['Lag_4days'] = gf_df['Growth_Factor'].shift(4)
    gf_df['Lag_5days'] = gf_df['Growth_Factor'].shift(5)
    gf_df['Lag_6days'] = gf_df['Growth_Factor'].shift(6)
    gf_df['Lag_7days'] = gf_df['Growth_Factor'].shift(7)
    gf_df['Days_since_03-13'] = np.arange(len(gf_df.index.tolist()))
    # Add date features.
    gf_df['month'] = gf_df.index.month
    gf_df['day'] = gf_df.index.day
    gf_df['day_week'] = gf_df.index.dayofweek
    # differenced features
    gf_df['Lag_1days_diff'] = gf_df['Lag_1days'].diff(1)
    return gf_df


class PredictGrowthFactor():
    """ Class to return predictions for growth factor by multiple
    estimators. No parameter, estimator validation done. Returns a dictionary with results
    for all the estimators.
    """

    def __init__(self):
        # initialising models
        self.lin_reg = LinearRegression()
        # poly reg - regularisation hyper-parameter alpha hardcoded from manual hyp. optim.
        self.pipe_gf = Pipeline([('poly', PolynomialFeatures(degree=2)),
                                 ('scale', StandardScaler()), ('ridge', Ridge(alpha=21))])

    def fit(self, x, y):
        self.x = x
        self.y = y

        # fit simple models (mean)
        self.mean = np.array(y).mean()

        # fitting the regression models
        self.lin_reg.fit(x, y)
        self.pipe_gf.fit(x, y)

    def predict(self, X):
        # dictionary for results.
        self.results = {}
        # storing predictions
        self.results['lin_reg'] = self.lin_reg.predict(X)
        self.results['ridge_reg_deg_2'] = self.pipe_gf.predict(X)
        self.results['last_month_mean'] = np.repeat(self.mean, len(X))

        return self.results
