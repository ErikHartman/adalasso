# Adaptive Lasso

A scikit-learn compatible implementation of the Adaptive Lasso algorithm for feature selection and regression.

## Overview

Adaptive Lasso is an extension of the standard Lasso method that provides improved feature selection properties through weighted L1 penalties. It assigns different weights to different coefficients in the L1 penalty, usually based on preliminary estimates of the coefficients.

This implementation follows the scikit-learn API design and can be used as a drop-in replacement for other scikit-learn linear models.

## Installation

```bash
pip install adalasso
```

## Usage

```python
from adalasso import AdaptiveLasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and fit the model
model = AdaptiveLasso(alpha=0.1, gamma=1.0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Access model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

## Parameters

- `alpha`: Regularization strength
- `gamma`: Power parameter for the adaptive weights
- `fit_intercept`: Whether to calculate the intercept
- `max_iter`: Maximum number of iterations
- `tol`: Tolerance for optimization

See the class documentation for a full list of parameters.

## Visualization Examples

### Regularization Path Comparison

The following figure shows how coefficients evolve with different regularization strengths for standard Lasso versus Adaptive Lasso:

![Adaptive Lasso Regularization Path](adaptive_lasso_regularization_path.png)

### Feature Selection Performance

This figure demonstrates how Adaptive Lasso performs better at selecting the correct number of non-zero features across different regularization strengths:

![Feature Selection vs Regularization](adaptive_lasso_feature_count.png)

## References

- Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of the American Statistical Association, 101(476), 1418-1429.

## License

MIT License
