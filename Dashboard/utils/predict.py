"""
All the machine learning, statistical models for the Covid Forecasting Dashboard.
"""
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor, GammaRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# curve fit function from scipy.optimize to fit a function using nonlinear least squares.
from scipy.optimize import curve_fit

# Time series functions

from statsmodels.tsa import stattools
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


class RegressionModelsGrowthFactor():
    """ Class to return predictions for growth factor by multiple
    estimators. Feature scaling done, hardcoded hyperparameters.
    No parameter, estimator validation done. Returns a dictionary with results
    for all the estimators. Allows Recursive multi-step forecast
    """

    def __init__(self, recursive_forecast=True):

        # optional parameters
        self.recursive_forecast = recursive_forecast

        # dictionary for results.
        self.results = {}

        # initialising models
        self.lin_reg = LinearRegression()

        # Ridge Reg with polynomials of 2nd order, hardcoded

        self.pipe_gf = Pipeline([('poly', PolynomialFeatures(degree=2)),
                                 ('scale', StandardScaler()), ('ridge', Ridge(alpha=1.0))])

        # default state of data None - Also used to check if model is fit.
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

        # fit simple models (mean)
        self.mean = np.array(y).mean()

        # fitting the regression models
        self.lin_reg.fit(x, y)
        self.pipe_gf.fit(x, y)

    def predict(self, X):
        # check if estimators fit - this isn't entirely correct, One could change x to a string
        # after defining an instance and the check would pass, check for exact class instance and not
        # what its not.
        if self.x is None or self.y is None:
            raise NotFittedError("Estimator instance is not fitted yet. Call 'fit'")

        ### Recursive Multi-Step Predictions if Lag features used ###

        if self.recursive_forecast:
            # print('Lagged features - multi-step forecast')
            self.x_test_lin = X.astype(np.float64).copy()
            self.x_test_ridge = X.astype(np.float64).copy()

            # faster to append to a list
            self.pred_list_lin = []
            self.pred_list_ridge = []

            # Recursively predicting next time-step and replacing back into test data
            for i in np.arange(X.shape[0]):
                self.preds_lin = self.lin_reg.predict(self.x_test_lin.iloc[0, :].values.reshape(1, -1))
                self.preds_ridge = self.pipe_gf.predict(self.x_test_ridge.iloc[0, :].values.reshape(1, -1))

                # saving forecast
                self.pred_list_lin.append(self.preds_lin)
                self.pred_list_ridge.append(self.preds_ridge)

                # shift the lagged values in test DF by 1
                self.x_test_lin = self.x_test_lin.shift(periods=-1)
                self.x_test_ridge = self.x_test_ridge.shift(periods=-1)

                # putting prediction back into first lag place for shifted df(day t+1)
                self.x_test_lin.iloc[0, 0] = self.preds_lin
                self.x_test_ridge.iloc[0, 0] = self.preds_ridge

            # saving predictions in Dict
            self.results['Linear_Regression'] = np.array(self.pred_list_lin).ravel()
            self.results['Ridge_Regression'] = np.array(self.pred_list_ridge).ravel()

            # If no lag features used
        elif not self.recursive_forecast:
            self.results['Linear_Regression'] = self.lin_reg.predict(X)
            self.results['Ridge_Regression'] = self.pipe_gf.predict(X)

        # storing baseline last month mean predictions
        self.results['Last_Month_Mean'] = np.repeat(self.mean, len(X))

        return self.results


class TimeSeriesGrowthFactor():
    """Sklearn Regression wrapper for SARIMA(1, 1, 1)x(0, 1, 1, 7) model used for predicting Growth Factor.
    Runs ADF test for the transformation and if significant only then does it allow a model to be fit.
    Else Error out.
    """

    def __init__(self):  # SARIMAX try with inheriting SARIMAX
        # Private Attribute To check if time-series is stationary and fitted.
        self.__valid = None

    def fit(self, ts):
        self.Y = ts

        # taking moving average to remove seasonality.
        self.__Y = self.Y.rolling(window=7).mean()[6:]
        # removing trend
        self.__Y = self.__Y.diff(periods=1)[1:]
        # performing ADF test
        self.adf_results = stattools.adfuller(self.__Y, maxlag=None, autolag='AIC')

        # check if p-value is less than 0.05 to reject presence of unit root
        # then fit model.
        if self.adf_results[1] > 0.05:
            self.__valid = 0
            raise NotFittedError(
                'Can\'t Fit estimator. Time-Series likely not stationary with current SARIMA parameters')

        # setup and fit model if we don't error out.
        self.model = SARIMAX(self.Y, trend='n', order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))

        # change status for stationary and fit.
        self.__valid = 1

        # result wrapper with all params and summary visible.
        self.res_model = self.model.fit(disp=0)

    def predict(self, X_start, X_end):
        """Pass in number of days since(X_start) and to(X_end) for which forecast is required.
        """
        if self.__valid is None:
            raise NotFittedError("Estimator instance is not fitted yet. Call 'fit'")

        return self.res_model.predict(start=X_start, end=X_end)


""" Growth Ratio Prediction Functions 
"""


class GrowthRatioFeatures(BaseEstimator, TransformerMixin):
    """Data transformation for easily creating Growth ratio features from the total confirmed cases.
    Total confirmed is a date indexed series with total confirmed covid cases. If GLM bounds is true,
    shifts growth ratio into the correct range [0,+inf] by subtracting 1.
    First calculates the growth ratio. Performs the following transformations.
    Lagged Feats - Lag of growth ratio for t days.
    Diffed Feats - 1st Differed growth ratio of t lags. Useful to encode trend information.
    Date Feats - used to create date feats like month, day, dayofweek to help encode seasonality.
    """

    def __init__(self, num_lagged_feats=7, num_diff_feats=1, date_feats=True, glm_bounds=True):
        self.num_lagged_feats_ = num_lagged_feats
        self.num_diff_feats_ = num_diff_feats
        self.date_feats_ = date_feats
        self.glm_bounds_ = glm_bounds
        self.X = None
        self.y = None

    def fit(self, X, y=None):
        # nothing to calculate for a transformation ex, calculating mean of data etc.
        # could add gr calculation here.
        return self

    def transform(self, X, y=None):
        # calculating growth ratio (target, y)
        self.y = X / X.shift(1)
        self.y = self.y.to_frame(name='Growth_Ratio')
        self.y.dropna(inplace=True)
        # subtracting 1 to get into right bounds.
        if self.glm_bounds_:
            self.y = self.y - 1

        # creating features (features, x) - important to create a copy of target here, otherwise overwrites y

        self.X = self.y.copy()

        # creating lagged features
        for i in range(1, self.num_lagged_feats_ + 1):
            self.X[f'Lag_{i}days'] = self.y.shift(i)

        # creating date features
        if self.date_feats_:
            self.X['month'] = self.X.index.month
            self.X['day'] = self.X.index.day
            self.X['day_week'] = self.X.index.dayofweek
            self.X[f'Days_since_{self.X.index.date.min()}'] = np.arange(len(self.X.index.tolist()))

            # creating differenced features
            # check to see if lagged features exist and num of diff features less than lagged feats.
            if (self.num_lagged_feats_ >= 1) & (self.num_lagged_feats_ >= self.num_diff_feats_):

                for i in range(1, self.num_diff_feats_ + 1):
                    self.X[f'Lag_{i}days_diff'] = self.X[f'Lag_{i}days'].diff(1)
            else:
                print('Number of diffed lag features requested higher than number of lagged features')

        # dropping growth ratio(target) from x
        self.X.drop('Growth_Ratio', axis=1, inplace=True)

        # if no features generated, pass just the days since feature as feature(can break)
        if self.X.shape[1] == 0:
            self.X[f'Days_since_{self.X.index.date.min()}'] = np.arange(len(self.X.index.tolist()))

        return self.X, self.y


def perf_adf(ts):
    """Performs ADF test on time series and prints Results.
    """
    dftest = stattools.adfuller(ts, maxlag=None, autolag='AIC')
    df_result = pd.Series(dftest[:4], index=['Test- Stat', 'P-Value', '# of Lags Used', '# of Obs Used'])
    # For Critical values at diff. confidence intervals
    for key, value in dftest[4].items():
        df_result[f'Critical Value {key}'] = value
    return df_result.head(10)


def train_test_split_gr(x, y, validation_days=30):
    """Return in form of numpy array ?"""
    # Removing all NaN values - not best to abstract this away
    index = x.isna().sum().max()
    # print(f'{index} samples (Days) dropped from start of feature vector due to nans')
    x_temp = x[index:]
    target = y[index:]
    # train-test split

    z = validation_days
    x_train = x_temp[:-z]
    x_test = x_temp[-z:]
    y_train = target[:-z]
    y_test = target[-z:]

    return x_train, x_test, y_train.values.ravel(), y_test.values.ravel()


class RegressionModelsGrowthRatio():
    """Class to return forecasts for growth ratio by several
    regression estimators :
    1.Linear Regression. 2. Poisson Regression. 3. Gamma Regression.
    Can accept lagged target as input features and provides recursive forecast in predict method.
    The lag features t-1, t-2,..., t-n must in the first n columns of the input feature matrix in order.
    args:
    correct_glm_bounds = bool; if passed corrects target growth ratio range to glm bounds [0, +\inf]
    recursive_forecast = bool; For letting the estimators know that lagged target features are being used
    and forecast needs to be done in a recursive approach where forecast for day t are used as features on day t+1.
    Contains separate Pipelines for each model with hardcoded parameters for polynomial features, scaling.
    Returns: a dictionary with results for all the estimators.

    """

    def __init__(self, correct_glm_bounds=True, recursive_forecast=False):

        # optional parameters
        self.correct_glm_bounds = correct_glm_bounds
        self.recursive_forecast = recursive_forecast

        # pipelines for the models.
        # Scaling for Poisson and Gamma Regression models, they use L2 regularization penalty
        self.pipe_lin_reg_ar = Pipeline([('poly', PolynomialFeatures(1, include_bias=False)),
                                         ('scale', StandardScaler()), ('reg_lin', LinearRegression())])
        self.pipe_reg_pois = Pipeline([('poly', PolynomialFeatures(2, include_bias=False)),
                                       ('scale', StandardScaler()),
                                       ('reg_pois', PoissonRegressor(alpha=0, max_iter=5000))])
        self.pipe_reg_gamm = Pipeline([('poly', PolynomialFeatures(2, include_bias=False)),
                                       ('scale', StandardScaler()),
                                       ('reg_gamm', GammaRegressor(alpha=0, max_iter=5000))])
        # initial data values for checking estimators fit ?
        self.x = None
        self.y = None
        self.x_ar = None
        self.y_ar = None
        # dictionary for results.
        self.results = {}

    def fit(self, x, y, x_ar, y_ar):
        """Takes in features and labels for both the regression and AR models.
        NOTE: Very sketchy way to do this. Ideally shouldn't even have two different 'class' of estimatorss together
        That fit on different data.
        """
        # assign data
        self.x = x
        self.y = y
        self.x_ar = x_ar
        self.y_ar = y_ar

        # fit models
        self.pipe_reg_pois.fit(x, y)
        self.pipe_reg_gamm.fit(x, y)
        self.pipe_lin_reg_ar.fit(x_ar, y_ar)

    def predict(self, X, X_ar):
        # check if estimators fit - this isn't entirely correct

        # iterables of rules
        rules = [self.x is None, self.y is None, self.x_ar is None, self.y_ar is None]
        # any - if any of them is true, returns true.
        if any(rules):
            raise NotFittedError("Estimator instance is not fitted yet. Call 'fit'")

            # storing predictions of AR model
        # print('Ar model preds')
        self.results['ARModel'] = self.pipe_lin_reg_ar.predict(X_ar)

        ### Recursive Multi-Step Predictions if Lag features used ###

        if self.recursive_forecast:
            # print('Lagged features - multi-step forecast')
            self.x_test_gam = X.astype(np.float64).copy()
            self.x_test_poi = X.astype(np.float64).copy()

            # faster to append to a list
            self.pred_list_gam = []
            self.pred_list_poi = []

            # Recursively predicting next time-step and replacing back into test data
            for i in np.arange(X.shape[0]):
                self.preds_gamm = self.pipe_reg_gamm.predict(self.x_test_gam.iloc[0, :].values.reshape(1, -1))
                self.preds_poi = self.pipe_reg_pois.predict(self.x_test_poi.iloc[0, :].values.reshape(1, -1))

                # saving forecast
                self.pred_list_gam.append(self.preds_gamm)
                self.pred_list_poi.append(self.preds_poi)

                # shift the lagged values in test DF by 1
                self.x_test_gam = self.x_test_gam.shift(periods=-1)
                self.x_test_poi = self.x_test_poi.shift(periods=-1)

                # putting prediction back into first lag place for shifted df(day t+1)
                self.x_test_gam.iloc[0, 0] = self.preds_gamm
                self.x_test_poi.iloc[0, 0] = self.preds_poi

            # saving predictions in Dict
            self.results['GammaReg'] = np.array(self.pred_list_gam).ravel()
            self.results['PoissonReg'] = np.array(self.pred_list_poi).ravel()

        # when no lag features used. Simple forecast
        elif self.recursive_forecast == False:
            # print('No Lagged features')
            self.results['PoissonReg'] = self.pipe_reg_pois.predict(X)
            self.results['GammaReg'] = self.pipe_reg_gamm.predict(X)

        # move back to actual bounds before correction for GLM
        if self.correct_glm_bounds:
            self.results['ARModel'] += 1
            self.results['PoissonReg'] += 1
            self.results['GammaReg'] += 1

        return self.results
