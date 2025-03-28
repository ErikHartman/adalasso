"""
==============================
Adaptive Lasso Regularization Path
==============================

This example illustrates the regularization path of Adaptive Lasso
compared to standard Lasso, showing how coefficients evolve as the 
regularization strength (α) changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from adalasso import AdaptiveLasso

np.random.seed(42)

n_samples, n_features = 100, 20
n_informative = 5
X, y, true_coef = make_regression(
    n_samples=n_samples, 
    n_features=n_features, 
    n_informative=n_informative,
    noise=1, 
    coef=True,
    random_state=42
)

true_nonzero_idx = np.where(np.abs(true_coef) > 1e-10)[0]
print(f"Truly informative features: {true_nonzero_idx}")
print(f"Their coefficients: {true_coef[true_nonzero_idx]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
alphas = np.logspace(-3, 4, 30)

paths = {
    'Lasso': [],
    'Adaptive Lasso (γ=1.0)': [],
    'Adaptive Lasso (γ=2.0)': []
}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
    lasso.fit(X_scaled, y)
    paths['Lasso'].append(lasso.coef_.copy())
    
    ada_lasso1 = AdaptiveLasso(alpha=alpha, gamma=1.0, max_iter=10000, tol=1e-4)
    ada_lasso1.fit(X_scaled, y)
    paths['Adaptive Lasso (γ=1.0)'].append(ada_lasso1.coef_.copy())
    
    ada_lasso2 = AdaptiveLasso(alpha=alpha, gamma=2.0, max_iter=10000, tol=1e-4)
    ada_lasso2.fit(X_scaled, y)
    paths['Adaptive Lasso (γ=2.0)'].append(ada_lasso2.coef_.copy())


for model in paths:
    paths[model] = np.array(paths[model])


plt.figure(figsize=(8, 8))

feature_colors = ['lightgray'] * n_features
distinct_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, feature_idx in enumerate(true_nonzero_idx):
    feature_colors[feature_idx] = distinct_colors[i % len(distinct_colors)]

plt.subplot(3, 1, 1)
for i in range(n_features):
    is_informative = i in true_nonzero_idx
    plt.semilogx(alphas, paths['Lasso'][:, i], color=feature_colors[i], 
                 alpha=0.7 if is_informative else 0.3,
                 linewidth=2 if is_informative else 1)

plt.xlabel('α (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regularization Path')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
for i in range(n_features):
    is_informative = i in true_nonzero_idx
    plt.semilogx(alphas, paths['Adaptive Lasso (γ=1.0)'][:, i], color=feature_colors[i],
                 alpha=0.7 if is_informative else 0.3,
                 linewidth=2 if is_informative else 1)

plt.xlabel('α (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Adaptive Lasso (γ=1.0) Regularization Path')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
for i in range(n_features):
    is_informative = i in true_nonzero_idx
    plt.semilogx(alphas, paths['Adaptive Lasso (γ=2.0)'][:, i], color=feature_colors[i],
                 alpha=0.7 if is_informative else 0.3,
                 linewidth=2 if is_informative else 1)

plt.xlabel('α (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Adaptive Lasso (γ=2.0) Regularization Path')
plt.grid(True, alpha=0.3)

plt.figlegend(
    [plt.Line2D([0], [0], color=feature_colors[idx], linewidth=2) for idx in true_nonzero_idx] +
    [plt.Line2D([0], [0], color='#1f77b4', linewidth=1, alpha=0.3)],
    [f'Feature {idx} (True Non-Zero)' for idx in true_nonzero_idx] + ['Other Features (True Zero)'],
    loc='center right', bbox_to_anchor=(1.3, 0.5), frameon=False
)

plt.tight_layout()
plt.savefig('adaptive_lasso_regularization_path.png', bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,4))

for model in paths:
    non_zero_counts = np.sum(np.abs(paths[model]) > 1e-6, axis=1)
    plt.semilogx(alphas, non_zero_counts, marker='o', linewidth=2, label=model)

plt.axhline(y=len(true_nonzero_idx), color='k', linestyle='--', alpha=0.7, 
            label=f'True Non-Zero Count ({len(true_nonzero_idx)})')

plt.xlabel('α (log scale)')
plt.ylabel('Number of Non-Zero Coefficients')
plt.title('Feature Selection vs. Regularization Strength')
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('adaptive_lasso_feature_count.png')
plt.show()
