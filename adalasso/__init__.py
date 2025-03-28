"""
The adaptive lasso is a modification of the standard lasso
where the penalty term is weighted by a preliminary estimate
of the coefficients.
"""

from ._adaptive_lasso import AdaptiveLasso

__all__ = ["AdaptiveLasso"]
