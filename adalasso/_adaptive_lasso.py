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
    initial_ridge_alpha : float, default=0.1
        Regularization strength for the initial Ridge regression.
    max_weight : float, default=100.0
        Maximum value for the adaptive weights to prevent numerical issues.
    weight_epsilon : float, default=1e-6
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
        random_state=None, initial_ridge_alpha=0.1, max_weight=100.0,
        weight_epsilon=1e-6
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
        """Fit adaptive lasso model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=False)

        # Handle normalization (deprecated in newer scikit-learn versions).
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

        # Center X and y manually if fit_intercept is True
        if self.fit_intercept:
            X_offset = np.average(X_processed, axis=0)
            y_offset = np.average(y, axis=0)
            # Center X and y
            X_processed = X_processed - X_offset
            y_processed = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0
            y_processed = y

        # For post-solution coefficient correction
        if self.normalize:
            X_scale = X_scaler.scale_
        else:
            X_scale = np.ones(X.shape[1])

        # Step 1: get initial weights using Ridge with stronger regularization
        initial_model = Ridge(alpha=self.initial_ridge_alpha, fit_intercept=False)
        initial_model.fit(X_processed, y_processed)
        init_coefs = np.abs(initial_model.coef_)
        
        # Apply a higher threshold to avoid extremely small values
        # that would produce huge weights
        init_coefs = np.maximum(init_coefs, self.weight_epsilon)
        
        # Step 2: compute the adaptive weights with capping to prevent explosion
        w = 1.0 / (init_coefs ** self.gamma)
        
        # Cap the weights to prevent extreme values
        w = np.minimum(w, self.max_weight)
        
        # Normalize the weights to keep the scale reasonable
        # This doesn't change the relative importance but prevents numerical issues
        if np.max(w) > 0:
            w = w / np.mean(w)  # Normalize so the average weight is 1
        
        # Apply the square-root trick with numerical stability
        sqrt_w = np.sqrt(w)
        X_weighted = X_processed / sqrt_w
        
        # Step 3: Fit the standard Lasso on the scaled data
        lasso_model = Lasso(
            alpha=self.alpha,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            precompute=self.precompute,
            copy_X=False,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state
        ).fit(X_weighted, y_processed)
        
        # The lasso model's coefficients are in the weighted space
        weighted_coefs = lasso_model.coef_
        # Convert back to the original space with controlled scaling
        self.coef_ = weighted_coefs / sqrt_w
        
        # Now set the intercept if required
        if self.fit_intercept:
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_ / X_scale)
        else:
            self.intercept_ = 0.0
            
        self.n_iter_ = lasso_model.n_iter_
        
        return self

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        return np.dot(X, self.coef_) + self.intercept_
