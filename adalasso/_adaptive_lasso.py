import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class AdaptiveLasso(BaseEstimator, RegressorMixin):
    """Adaptive Lasso linear model.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term.
    gamma : float, default=1.0
        Power parameter for adaptive weights.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Tolerance for optimization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    initial_ridge_alpha : float, default=1.0
        Ridge penalty strength for initial coefficient estimates.
    max_weight : float, default=100.0
        Maximum value for adaptive weights to prevent numerical issues.
    weight_epsilon : float, default=1e-6
        Small value to avoid division by zero in adaptive weights.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term in decision function.
    n_iter_ : int
        Number of iterations run by coordinate descent solver.
    """

    def __init__(
        self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4,
        fit_intercept=True, initial_ridge_alpha=1.0, 
        max_weight=100.0, weight_epsilon=1e-6, random_state=None
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.initial_ridge_alpha = initial_ridge_alpha
        self.max_weight = max_weight
        self.weight_epsilon = weight_epsilon
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=False)

        # Center X and y if intercept is used
        if self.fit_intercept:
            X_offset = np.mean(X, axis=0)
            y_offset = np.mean(y)
            X_centered = X - X_offset
            y_centered = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0
            X_centered = X
            y_centered = y

        # Step 1: Obtain initial coefficient estimates using Ridge regression
        ridge = Ridge(alpha=self.initial_ridge_alpha, fit_intercept=False)
        ridge.fit(X_centered, y_centered)
        initial_coefs = np.abs(ridge.coef_)

        # Avoid extremely small values
        initial_coefs = np.maximum(initial_coefs, self.weight_epsilon)

        # Step 2: Compute adaptive weights
        weights = 1.0 / (initial_coefs ** self.gamma)

        # Cap adaptive weights to prevent numerical instability
        weights = np.minimum(weights, self.max_weight)

        # (Optional) Normalize weights for numerical stability
        weights /= np.mean(weights)

        # Transform features according to adaptive weights
        X_weighted = X_centered / np.sqrt(weights)

        # Step 3: Fit standard Lasso to transformed features
        lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        lasso.fit(X_weighted, y_centered)

        # Convert coefficients back to original space
        self.coef_ = lasso.coef_ / np.sqrt(weights)

        # Intercept computation
        if self.fit_intercept:
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_)
        else:
            self.intercept_ = 0.0

        self.n_iter_ = lasso.n_iter_

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        return X @ self.coef_ + self.intercept_
