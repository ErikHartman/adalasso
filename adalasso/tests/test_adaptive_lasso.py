import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from adalasso import AdaptiveLasso

def test_adaptive_lasso_basic():
    """Test basic functionality of AdaptiveLasso."""
    # Generate synthetic data with known sparse structure
    X, y = make_regression(
        n_samples=100, 
        n_features=20, 
        n_informative=5, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit model
    model = AdaptiveLasso(alpha=0.1, gamma=1.0)
    model.fit(X_train, y_train)
    
    # Basic sanity checks
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    assert len(model.coef_) == X.shape[1]
    
    # Check predictions work
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
    
    # Check reasonable performance (R² > 0)
    assert r2_score(y_test, y_pred) > 0

def test_adaptive_lasso_sparsity():
    """Test that AdaptiveLasso produces sparse solutions."""
    # Generate data
    X, y = make_regression(
        n_samples=100, 
        n_features=20, 
        n_informative=5, 
        random_state=42
    )
    
    # Try different alpha values to check sparsity
    alphas = [0.01, 0.1, 1.0, 10.0]
    
    for alpha in alphas:
        model = AdaptiveLasso(alpha=alpha, gamma=1.0)
        model.fit(X, y)
        
        # Higher alpha should give more sparsity (more zeros)
        # Just check that we have some zeros
        assert np.sum(model.coef_ == 0) > 0

def test_adaptive_lasso_parameters():
    """Test that different parameter values work."""
    # Generate data
    X, y = make_regression(
        n_samples=100, 
        n_features=10, 
        n_informative=5, 
        random_state=42
    )
    
    # Try different parameter combinations
    for alpha in [0.1, 1.0]:
        for gamma in [0.5, 1.0, 2.0]:
            for fit_intercept in [True, False]:
                model = AdaptiveLasso(
                    alpha=alpha, 
                    gamma=gamma,
                    fit_intercept=fit_intercept
                )
                model.fit(X, y)
                
                # Just make sure it runs without errors
                assert hasattr(model, 'coef_')
