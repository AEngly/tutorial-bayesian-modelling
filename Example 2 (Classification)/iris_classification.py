"""
Iris Dataset Classification with Bayesian Methods
================================================

This script demonstrates Bayesian classification using the famous Iris dataset.
It includes:
1. Data loading and exploration
2. Feature visualization with different species highlighted
3. Bayesian logistic regression for classification
4. Model evaluation and visualization

Authors: Andreas Engly (ANENG) and Anders Runge Walther (AWR)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm
from matplotlib.patches import Ellipse

# Set random seeds for reproducibility
numpyro.set_platform("cpu")
np.random.seed(42)

# =============================================================================
# DATA LOADING AND EXPLORATION
# =============================================================================

def load_iris_data():
    """
    Load the Iris dataset and return features and target.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (sepal length, sepal width, petal length, petal width).
    y : array-like, shape (n_samples,)
        Target labels (0: setosa, 1: versicolor, 2: virginica).
    feature_names : list
        Names of the features.
    target_names : list
        Names of the target classes.
    """
    print("Loading Iris dataset...")

    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y, feature_names, target_names

# Load the data
X, y, feature_names, target_names = load_iris_data()

# =============================================================================
# FEATURE VISUALIZATION
# =============================================================================

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a confidence ellipse for a 2D dataset.

    Parameters
    ----------
    x, y : array-like
        Data points for the ellipse.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse onto.
    n_std : float
        The number of standard deviations to determine the ellipse's radius.
    facecolor : str
        Color of the ellipse face.
    **kwargs
        Additional arguments passed to the Ellipse constructor.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate covariance matrix
    cov = np.cov(x, y)

    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Order eigenvalues and eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # Calculate angle of rotation
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    # Calculate width and height of ellipse
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # Create ellipse
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                     facecolor=facecolor, **kwargs)

    return ax.add_patch(ellipse)

def create_clustering_plot(X, y, feature_names, target_names):
    """
    Create a simple 2-subplot figure showing clear vs poor clustering with confidence ellipses.

    This function creates exactly 2 plots:
    1. Clear clustering: Petal Length vs Petal Width (excellent separation)
    2. Poor clustering: Sepal Length vs Sepal Width (overlapping species)
    """
    print("Creating clustering comparison plot with confidence ellipses...")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for each species
    colors = ['red', 'green', 'blue']
    ellipse_colors = ['lightcoral', 'lightgreen', 'lightblue']

    # Plot 1: Clear clustering - Petal Length vs Petal Width
    # This shows excellent separation between species
    for i, species in enumerate(target_names):
        mask = y == i
        x_data = X[mask, 2]  # Petal Length
        y_data = X[mask, 3]  # Petal Width

        # Plot data points
        ax1.scatter(x_data, y_data, label=species, alpha=0.8, s=60,
                   color=colors[i], edgecolors='black', linewidth=0.5)

        # Add confidence ellipse (95% confidence interval)
        confidence_ellipse(x_data, y_data, ax1, n_std=2.0,
                          edgecolor=colors[i], linewidth=2, alpha=0.3,
                          facecolor=ellipse_colors[i])

    ax1.set_xlabel('Petal Length [cm]')
    ax1.set_ylabel('Petal Width [cm]')
    ax1.set_title('Clear Clustering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Poor clustering - Sepal Length vs Sepal Width
    # This shows overlapping species (especially versicolor and virginica)
    for i, species in enumerate(target_names):
        mask = y == i
        x_data = X[mask, 0]  # Sepal Length
        y_data = X[mask, 1]  # Sepal Width

        # Plot data points
        ax2.scatter(x_data, y_data, label=species, alpha=0.8, s=60,
                   color=colors[i], edgecolors='black', linewidth=0.5)

        # Add confidence ellipse (95% confidence interval)
        confidence_ellipse(x_data, y_data, ax2, n_std=2.0,
                          edgecolor=colors[i], linewidth=2, alpha=0.3,
                          facecolor=ellipse_colors[i])

    ax2.set_xlabel('Sepal Length [cm]')
    ax2.set_ylabel('Sepal Width [cm]')
    ax2.set_title('Poor Clustering')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iris_clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Saved plot: iris_clustering_comparison.png")

def create_correlation_plot(X, feature_names, target_names):
    """
    Create a correlation heatmap showing feature relationships.
    """
    print("Creating correlation heatmap...")

    # Create DataFrame for easier correlation calculation
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]

    # Calculate correlation matrix
    correlation_matrix = df[feature_names].corr()

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation')

    # Update axis labels to include units
    feature_names_with_units = [name.replace('(cm)', '[cm]') for name in feature_names]
    plt.xticks(range(len(feature_names)), feature_names_with_units, rotation=45)
    plt.yticks(range(len(feature_names)), feature_names_with_units, rotation=0)

    plt.tight_layout()
    plt.savefig('iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Saved plot: iris_correlation_heatmap.png")

# Create the clustering comparison plot
create_clustering_plot(X, y, feature_names, target_names)

# Create the correlation plot
create_correlation_plot(X, feature_names, target_names)

# =============================================================================
# DATA PREPARATION FOR BAYESIAN CLASSIFICATION
# =============================================================================

def prepare_data_for_classification(X, y, test_size=0.5, random_state=42):
    """
    Prepare data for Bayesian classification.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Training and testing sets.
    """
    print(f"\nPreparing data for classification (test_size={test_size})...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test

# Prepare the data
X_train, X_test, y_train, y_test = prepare_data_for_classification(X, y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Build a Bayesian model to infer the number of clusters in the data
def bayesian_softmax_regression(X, y=None):
    """
    Bayesian multinomial logistic regression with correlated weights.
    """
    N, D = X.shape
    C = 3  # number of classes

    # Prior covariance for coefficients (can tune scale)
    cov_prior = jnp.eye(D) * 1.0  # D x D covariance

    # Beta: correlated across features within each class
    beta = numpyro.sample(
        "beta",
        dist.MultivariateNormal(jnp.zeros(D), covariance_matrix=cov_prior).expand([C])
    )  # shape: C x D

    # Intercept can remain independent
    intercept = numpyro.sample("intercept", dist.Normal(jnp.zeros(C), 1.0))

    # Compute logits and probabilities
    logits = jnp.dot(X, beta.T) + intercept
    probs = jax.nn.softmax(logits, axis=-1)

    # Observed labels
    numpyro.sample("obs", dist.Categorical(probs=probs), obs=y)

# =============================================================================
# RUN MCMC
# =============================================================================
def run_mcmc(X_train, y_train):
    numpyro.set_host_device_count(1)
    kernel = NUTS(bayesian_softmax_regression)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=1, progress_bar=True)
    rng_key = random.PRNGKey(42)
    print("Running Bayesian softmax regression model...")
    mcmc.run(rng_key, X_train, y_train)
    samples = mcmc.get_samples()
    print("MCMC finished!")
    return samples

# =============================================================================
# POSTERIOR PREDICTIVE FUNCTION
# =============================================================================
def posterior_predictive_probs(samples, X):
    """
    Compute posterior predictive class probabilities.
    """
    beta_samples = samples["beta"]       # shape: (num_samples, C, D)
    intercept_samples = samples["intercept"]  # shape: (num_samples, C)

    # logits: (num_samples, N_test, C)
    logits = jnp.einsum('sCD,ND->sNC', beta_samples, X) + intercept_samples[:, None, :]
    probs = jax.nn.softmax(logits, axis=-1)
    return probs

# =============================================================================
# MAIN WORKFLOW
# =============================================================================
# Example: load Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Prepare train/test split
X_train, X_test, y_train, y_test = prepare_data_for_classification(X, y)

# Run MCMC
samples = run_mcmc(X_train, y_train)

# Posterior predictive probabilities
probs_samples = posterior_predictive_probs(samples, X_test)
probs_mean = jnp.mean(probs_samples, axis=0)  # mean over MCMC samples
pred_labels = jnp.argmax(probs_mean, axis=-1)
accuracy = jnp.mean(pred_labels == y_test)
print(f"Test set accuracy: {accuracy:.3f}")

# =============================================================================
# PLOT POSTERIOR MEAN COEFFICIENTS WITH CREDIBLE INTERVALS
# =============================================================================
beta_samples = samples["beta"]  # shape: (num_samples, C, D)
beta_mean = jnp.mean(beta_samples, axis=0)
beta_p2_5 = jnp.percentile(beta_samples, 2.5, axis=0)
beta_p97_5 = jnp.percentile(beta_samples, 97.5, axis=0)

feature_names = ['Sepal Length [cm]', 'Sepal Width [cm]', 'Petal Length [cm]', 'Petal Width [cm]']
class_names = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(12, 6))
x = np.arange(len(feature_names))
width = 0.25

for c in range(3):
    # Calculate error bars (distance from mean to percentiles)
    yerr_lower = beta_mean[c] - beta_p2_5[c]
    yerr_upper = beta_p97_5[c] - beta_mean[c]

    plt.bar(x + c*width, beta_mean[c], width=width, alpha=0.7,
            label=class_names[c], yerr=[yerr_lower, yerr_upper],
            capsize=5, error_kw={'linewidth': 1})

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Bayesian Regression Coefficients (95% CI)")
plt.xticks(x + width, feature_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bayesian_softmax_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot: bayesian_softmax_coefficients.png")

# =============================================================================
# PLOT INTERCEPTS WITH CREDIBLE INTERVALS
# =============================================================================
intercept_samples = samples["intercept"]  # shape: (num_samples, C)
intercept_mean = jnp.mean(intercept_samples, axis=0)
intercept_p2_5 = jnp.percentile(intercept_samples, 2.5, axis=0)
intercept_p97_5 = jnp.percentile(intercept_samples, 97.5, axis=0)

plt.figure(figsize=(8, 6))
x = np.arange(len(class_names))
width = 0.6

# Calculate error bars (distance from mean to percentiles)
yerr_lower = intercept_mean - intercept_p2_5
yerr_upper = intercept_p97_5 - intercept_mean

plt.bar(x, intercept_mean, width=width, alpha=0.7,
        yerr=[yerr_lower, yerr_upper], capsize=5,
        error_kw={'linewidth': 1}, color=['red', 'green', 'blue'])

plt.xlabel("Classes")
plt.ylabel("Intercept Value")
plt.title("Bayesian Regression Intercepts (95% CI)")
plt.xticks(x, class_names)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bayesian_softmax_intercepts.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot: bayesian_softmax_intercepts.png")

# =============================================================================
# POSTERIOR PREDICTIVE CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(y_test, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig("bayesian_softmax_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot: bayesian_softmax_confusion_matrix.png")

# =============================================================================
# PRECISION, RECALL, F1 PLOT
# =============================================================================
precision, recall, f1, support = precision_recall_fscore_support(y_test, pred_labels, labels=[0, 1, 2])
metrics = {"Precision": precision, "Recall": recall, "F1-score": f1}

plt.figure(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.25

for i, (name, values) in enumerate(metrics.items()):
    plt.bar(x + i*width, values, width=width, alpha=0.7, label=name)

plt.xlabel("Classes")
plt.ylabel("Score")
plt.title("Classification Metrics")
plt.xticks(x + width, class_names)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bayesian_softmax_classification_metrics.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved plot: bayesian_softmax_classification_metrics.png")