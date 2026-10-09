"""The three price-forecast models used in Stage 1."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def scaled_svr(C=0.1):
    """SVR with scaled inputs and a scaled target (scalers are fitted on training data only)."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=0.1)),
        transformer=StandardScaler(),
    )


def make_models():
    """Return a fresh, unfitted copy of each model. Call once per fold."""
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=0),
        "SVR": scaled_svr(0.1),
    }