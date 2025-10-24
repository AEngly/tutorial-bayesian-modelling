import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import load_iris
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from jax.scipy.stats import multivariate_normal

# Load Iris dataset
data = load_iris()
X = data['data']
y = data['target']
class_names = data['target_names']
colors = ['blue', 'red', 'green']

# ----------------------------------------------------------------
# ############    Part 1: Visualization  #########################
# ----------------------------------------------------------------

# Choose feature pairs to compare
feature_pairs = [
    (2, 3),  # Petal length vs width
    (0, 1)   # Sepal length vs width
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (i, j) in zip(axes, feature_pairs):
    for c in range(3):
        Xc = X[y == c][:, [i, j]]
        ax.scatter(*Xc.T, label=class_names[c], alpha=0.5, color=colors[c])
        
        # Covariance ellipse
        mean = Xc.mean(axis=0)
        cov = np.cov(Xc.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)

        ell = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor=colors[c], facecolor='none', linewidth=2, linestyle='--'
        )
        ax.add_patch(ell)
    
    ax.set_xlabel(data['feature_names'][i])
    ax.set_ylabel(data['feature_names'][j])
    ax.grid(True)
    ax.axis('equal')

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=12)

plt.suptitle("  ", fontsize=16, y=1.07)
plt.tight_layout()
plt.savefig("./Example 2 (Classification)/iris_feature_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------
# ############           Part 2: Model             ###############
# ----------------------------------------------------------------

# --- Load and normalize data ---
data = load_iris()
X = data['data']
y = data['target']

X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
N, D = X_norm.shape
K_max = 5


# --- Define the GMM model ---
def gmm_model(data, K_max=8):
    N, D = data.shape

    with numpyro.plate("components", K_max):
        locs = numpyro.sample("locs", dist.Normal(0, 5).expand([K_max, D]))
        scales = numpyro.sample("scales", dist.HalfNormal(1.0).expand([K_max, D]))

    weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(K_max) * 0.3))

    with numpyro.plate("data", N):
        assignment = numpyro.sample("assignment", dist.Categorical(weights))
        numpyro.sample("obs", dist.MultivariateNormal(
            locs[assignment],
            covariance_matrix=jnp.diag(scales[assignment] ** 2)
        ), obs=data)


# --- Run MCMC inference ---
rng_key = random.PRNGKey(0)
kernel = NUTS(gmm_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(rng_key, data=X_norm, K_max=K_max)
samples = mcmc.get_samples()


# --- Compute posterior mean parameters ---
locs_post = samples['locs'].mean(axis=0)       # (K_max, D)
scales_post = samples['scales'].mean(axis=0)   # (K_max, D)
weights_post = samples['weights'].mean(axis=0) # (K_max,)

# weights_post is shape (K_max,)
weights_post_np = np.array(weights_post)  # convert JAX array to numpy
plt.figure(figsize=(8, 5))
plt.bar(range(K_max), weights_post_np, color='skyblue')
plt.xlabel("Cluster index")
plt.ylabel("Posterior mean weight")
plt.title("Posterior mean cluster weights")
plt.xticks(range(K_max))
plt.show()