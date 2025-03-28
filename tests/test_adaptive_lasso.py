import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class AdaptiveLasso(BaseEstimator, RegressorMixin):
    """Adaptive Lasso linear model with iterative fitting.

    The adaptive lasso is a modification of the standard lasso
    where the penalty term is weighted by a preliminary estimate
    of the coefficients.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term.
    gamma : float, default=1.0
        Power parameter for the adaptive weights.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Tolerance for optimization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
    precompute : bool, default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
    initial_ridge_alpha : float, default=10.0
        Regularization strength for the initial Ridge regression.
    max_weight : float, default=100.0
        Maximum value for the adaptive weights to prevent numerical issues.
    weight_epsilon : float, default=1e-4
        Small value to avoid division by zero in the adaptive weights.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Parameter vector (w in the cost function formula).
    intercept_ : float
        Independent term in decision function.
    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    def __init__(
        self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4,
        fit_intercept=True, normalize=False, precompute=False,
        copy_X=True, warm_start=False, positive=False,
        random_state=None, initial_ridge_alpha=10.0, max_weight=100.0,
        weight_epsilon=1e-4
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.initial_ridge_alpha = initial_ridge_alpha
        self.max_weight = max_weight
        self.weight_epsilon = weight_epsilon

    def fit(self, X, y):
        """Fit adaptive lasso model."""
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=False)

        # Optionally normalize (deprecated in new sklearns)
        if self.normalize:
            warnings.warn(
                "The normalize parameter is deprecated in scikit-learn and "
                "will be removed in a future version. Use StandardScaler instead.",
                FutureWarning
            )
            X_scaler = StandardScaler(with_mean=self.fit_intercept).fit(X)
            X_processed = X_scaler.transform(X)
        else:
            X_scaler = None
            X_processed = X.copy() if self.copy_X else X

        # Center X and y if needed
        if self.fit_intercept:
            X_offset = np.average(X_processed, axis=0)
            y_offset = np.average(y, axis=0)
            X_processed = X_processed - X_offset
            y_processed = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0
            y_processed = y

        if self.normalize:
            X_scale = X_scaler.scale_
        else:
            X_scale = np.ones(X.shape[1])

        # 1. Initial (Ridge) estimate
        initial_model = Ridge(alpha=self.initial_ridge_alpha, fit_intercept=False)
        initial_model.fit(X_processed, y_processed)
        init_coefs = np.abs(initial_model.coef_)

        # 2. Avoid zero denominators
        init_coefs = np.maximum(init_coefs, self.weight_epsilon)

        # 3. Compute w_j = 1 / |coef_j|^gamma
        w = 1.0 / (init_coefs ** self.gamma)

        # 4. Cap the weights
        w = np.minimum(w, self.max_weight)

        # (Optionally, if you DO want re-scaling by mean, do it here,
        #  then re-cap again. But let's skip that for now.)

        # 5. Square-root trick
        sqrt_w = np.sqrt(w)
        X_weighted = X_processed * sqrt_w

        # 6. Solve standard Lasso on scaled data
        lasso_model = Lasso(
            alpha=self.alpha,
            fit_intercept=False,  # Already centered
            max_iter=self.max_iter,
            tol=self.tol,
            precompute=self.precompute,
            copy_X=False,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state
        ).fit(X_weighted, y_processed)

        # 7. Transform back to original space
        weighted_coefs = lasso_model.coef_
        self.coef_ = weighted_coefs / sqrt_w

        # Intercept
        if self.fit_intercept:
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_ / X_scale)
        else:
            self.intercept_ = 0.0

        # Iterations used
        self.n_iter_ = lasso_model.n_iter_

        return self

    def predict(self, X):
        """Predict with the linear model."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        return np.dot(X, self.coef_) + self.intercept_
