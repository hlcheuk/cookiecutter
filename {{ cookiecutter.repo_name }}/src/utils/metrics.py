from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from math import sqrt

# bshishov/forecasting_metrics.py
# Reference: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9

#####################################
###                               ###
### Quick private functions       ###
###                               ###
#####################################


def _error(y_true: np.ndarray, y_pred: np.ndarray):
    """Simple error"""
    return y_true - y_pred


def _percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """Percentage error. Result is NOT multiplied by 100"""
    return _error(y_true, y_pred) / (y_true)


def _convert_np(y_true: np.ndarray, y_pred: np.ndarray):
    """Convert input to np array"""
    return (np.array(y_true), np.array(y_pred))


def _filter_by_epsilon(y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e3):

    """Define custom function to filter out the list with the value less than epsilon"""

    # CUSTOM: y_true >= epsilon only
    idx = np.where(y_true >= epsilon)[0]  # np.where returns tuple
    if len(idx) == 0:
        return np.nan
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    return (y_true, y_pred)


#####################################
###                               ###
### Evaluation metrics            ###
###                               ###
#####################################


def rmse(y_true, y_pred):
    """Wrapper for rmse in sklearn"""
    metric = sqrt(mean_squared_error(y_true, y_pred))
    return metric


def rsq(y_true, y_pred):
    """Wrapper for r2_score in sklearn"""
    metric = r2_score(y_true, y_pred, sample_weight=None, multioutput="uniform_average")
    return metric


def mae(y_true, y_pred):
    metric = mean_absolute_error(y_true, y_pred)
    return metric


def mape(y_true, y_pred, epsilon=1e3):
    """Custom mape function. Check with actual sales >= epsilon only"""

    y_true, y_pred = _convert_np(y_true, y_pred)
    y_true, y_pred = _filter_by_epsilon(y_true, y_pred, epsilon)
    metric = np.mean(np.abs(_percentage_error(y_true, y_pred))) * 100

    return metric


def wmape(y_true, y_pred, epsilon=1e3):
    """
    Weighted mean absolute percentage error

    Reference:
        https://ibf.org/knowledge/glossary/weighted-mean-absolute-percentage-error-wmape-299
    """

    y_true, y_pred = _convert_np(y_true, y_pred)
    y_true, y_pred = _filter_by_epsilon(y_true, y_pred, epsilon)

    upper = np.sum(np.abs(_percentage_error(y_true, y_pred)) * 100 * y_true)
    lower = np.sum(y_true)

    return upper / lower


def mdae(y_true, y_pred):
    """Median Absolute Error"""
    y_true, y_pred = _convert_np(y_true, y_pred)
    return np.median(np.abs(_error(y_true, y_pred)))


####
# Less usual
####


def me(y_true, y_pred):
    """Mean Error"""
    y_true, y_pred = _convert_np(y_true, y_pred)
    return np.mean(_error(y_true, y_pred))


def mpe(y_true, y_pred, epsilon=1e3):
    """Mean Percentage Error"""
    y_true, y_pred = _convert_np(y_true, y_pred)
    y_true, y_pred = _filter_by_epsilon(y_true, y_pred, epsilon)

    return np.mean(_percentage_error(y_true, y_pred)) * 100


def pctl_e(y_true, y_pred):

    """Return percentile of the simple error"""

    y_true, y_pred = _convert_np(y_true, y_pred)
    metric = np.percentile(
        a=_error(y_true, y_pred),  # Array
        q=[0, 25, 50, 75, 100],  # Quantile
        interpolation="linear",
    )

    # Rounding and to list
    res = [round(e, 2) for e in metric.tolist()]

    return res


def pctl_pe(y_true, y_pred, epsilon=1e3):

    """Return percentile of the custom percentage error"""

    y_true, y_pred = _convert_np(y_true, y_pred)
    y_true, y_pred = _filter_by_epsilon(y_true, y_pred, epsilon)

    metric = np.percentile(
        a=_percentage_error(y_true, y_pred),  # Array
        q=[0, 25, 50, 75, 100],  # Quantile
        interpolation="linear",
    )
    # In percentage
    metric = metric * 100
    res = [round(e, 2) for e in metric.tolist()]

    return res


####
# Test
####


def test_func():
    y_true = [2000, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    y_pred = [0, 1000, 800, 1000, 1000, 1000, 1000, 1000]
    epsilon = 1000

    y_true, y_pred = _convert_np(y_true, y_pred)
    y_true, y_pred = _filter_by_epsilon(y_true, y_pred, epsilon)
    a = _percentage_error(y_true, y_pred)

    arr = [
        2000 / 2000,
        -200 / 1000,
        100 / 1100,
        200 / 1200,
        300 / 1300,
        400 / 1400,
        500 / 1500,
    ]

    res = pctl_pe(y_true, y_pred, epsilon)
    ans = np.percentile(a=arr, q=[0, 25, 50, 75, 100], interpolation="linear")
    ans = ans * 100
    ans = [round(e, 2) for e in ans.tolist()]

    return res


if __name__ == "__main__":
    a = test_func()
    print(a)
