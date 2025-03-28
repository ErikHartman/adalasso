import numpy as np
from ._adaptive_lasso import AdaptiveLasso

def adalasso_path(
    X,
    y,
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    gamma=1.0,
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    warm_start=False,
    positive=False,
    random_state=None,
    initial_ridge_alpha=0.1,
    max_weight=100.0,
    weight_epsilon=1e-6
):
    """
    Compute the Adaptive Lasso path by iterating over a sequence of alphas.
    This mimics the scikit-learn style for lasso_path.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target values.
    eps : float, default=1e-3
        Length of the path. alpha_min / alpha_max = eps.
    n_alphas : int, default=100
        Number of alphas along the regularization path.
    alphas : array-like, default=None
        List of alphas to use. If None, alphas are set automatically
        on a log scale from alpha_max down to alpha_max * eps.
    gamma : float, default=1.0
        Power parameter for the adaptive weights in your AdaptiveLasso.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Tolerance for the stopping criterion.
    fit_intercept : bool, default=True
        Whether or not to fit an intercept.
    normalize : bool, default=False
        Ignored unless fit_intercept=True. If True, X is normalized.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    warm_start : bool, default=False
        Reuse the solution of the previous call to fit as initialization.
    positive : bool, default=False
        When True, forces coefficients to be positive.
    random_state : int or None, default=None
        Pseudo random number generator seed control.
    initial_ridge_alpha : float, default=0.1
        Ridge alpha used for the initial weighting step.
    max_weight : float, default=100.0
        Maximum value of the adaptive weights (to avoid numerical issues).
    weight_epsilon : float, default=1e-6
        Floor for the adaptive weights computation (avoid division by zero).

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The array of alphas used.
    coefs : ndarray of shape (n_features, n_alphas)
        Coefficients along the path. coefs[:, i] corresponds to
        the alpha at alphas[i].
    dual_gaps : list of length n_alphas
        Here we return a list of None (placeholders),
        since the current AdaptiveLasso implementation
        does not compute dual gaps.
    """

    # -----------------------------------------------------------------------
    # 1) If no alpha grid is given, generate one similar to how lasso_path does
    n_samples = X.shape[0]
    if alphas is None:
        # alpha_max for standard Lasso is the smallest alpha that sets all coefs to zero
        # i.e., alpha_max = (1/n_samples) * max|X^T y|.
        # We can use the same approach here as a starting point.
        alpha_max = np.max(np.abs(X.T @ y)) / (n_samples)
        if alpha_max < np.finfo(float).eps:
            alpha_max = 1e-3  # fallback
        alpha_min = eps * alpha_max
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]  # ensure descending

    # Prepare storage for coefficients
    n_features = X.shape[1]
    coefs = np.empty((n_features, len(alphas)), dtype=float)
    dual_gaps = [None] * len(alphas)  # placeholders, as we're not computing them

    # -----------------------------------------------------------------------
    # 2) For each alpha in the grid, fit a new AdaptiveLasso and store the coefficients
    # Adjust if you need a different import path or put `AdaptiveLasso` in the same file.

    for i, alpha_ in enumerate(alphas):
        model = AdaptiveLasso(
            alpha=alpha_,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            initial_ridge_alpha=initial_ridge_alpha,
            max_weight=max_weight,
            weight_epsilon=weight_epsilon,
        )

        # If you want to pass the previous solution as a warm start, you'd have to
        # modify the AdaptiveLasso class to respect an externally set coef_.
        # The provided class doesn't do that out-of-the-box.
        # model.coef_ = previous_coef

        model.fit(X, y)
        coefs[:, i] = model.coef_

        # Save the current coef if you want to pass it as warm start to the next iteration.
        previous_coef = model.coef_

    return alphas, coefs, dual_gaps
