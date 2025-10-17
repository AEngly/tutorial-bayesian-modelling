import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import beta, gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# --- Settings ---
true_p = 0.70
prior_alpha = 50
prior_beta = 50
max_samples = 4050
step_size = 50

# --- Pre-generate all flips ---
rng_key = random.PRNGKey(0)
all_flips = dist.Bernoulli(probs=true_p).sample(rng_key, (max_samples,))

# --- NumPyro model ---
def coin_model(flips=None):
    p = numpyro.sample("p", dist.Beta(prior_alpha, prior_beta))
    with numpyro.plate("data", len(flips)):
        numpyro.sample("obs", dist.Bernoulli(probs=p), obs=flips)

# --- Inference runner ---
def run_inference(flips):
    kernel = NUTS(coin_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=False)
    mcmc.run(random.PRNGKey(1), flips=flips)
    return mcmc.get_samples()["p"]

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 1, 2050)

def init_plot():
    ax.set_title("Posterior Distribution of p")
    ax.set_xlabel("p (coin bias)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 30)
    ax.grid(True)
    return []

# --- Get confidence intervals for prior ---
p_prior_lower_P025 = beta.ppf(0.025, prior_alpha, prior_beta)
p_prior_upper_P975 = beta.ppf(0.975, prior_alpha, prior_beta)

# --- Animation function ---
def animate(frame):

    # Determine number of samples to use
    n = 50 + frame * step_size
    if n > max_samples:
        ani.event_source.stop()
        return []

    flips_so_far = all_flips[:n]
    posterior = run_inference(flips_so_far)
    p_posterior = np.mean(posterior)
    p_lower_P025 = np.percentile(posterior, 2.5)
    p_upper_P975 = np.percentile(posterior, 97.5)

    # --- Clear plot ---
    ax.clear()

    # --- Posterior KDE ---
    kde = gaussian_kde(posterior)
    y_post = kde(x)
    ax.fill_between(x, y_post, alpha=0.5, color='skyblue', label="Posterior Distribution (KDE)")
    ax.plot(x, y_post, color='blue')

    # --- Prior PDF ---
    y_prior = beta.pdf(x, prior_alpha, prior_beta)
    ax.fill_between(x, y_prior, alpha=0.3, color='orange', label="Prior Distribution (Beta)")
    ax.plot(x, y_prior, 'orange', linestyle='--')

    # --- Vertical lines ---
    ax.axvline(true_p, color='red', linestyle='--', label=f"True p = {true_p:.2f}")
    ax.axvline(p_posterior, color='blue', linestyle='--', label=f"Posterior p = {p_posterior:.2f} [{p_lower_P025:.2f}, {p_upper_P975:.2f}]")
    ax.axvline(0.50, color='gray', linestyle=':', label=f"Prior p = 0.50 [{p_prior_lower_P025:.2f}, {p_prior_upper_P975:.2f}]")

    # --- Axis titles and limits ---
    ax.set_title(f"Posterior after {n} flips")
    ax.set_xlabel("p (coin bias)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 30)
    ax.grid(True)
    ax.legend()

    return []

# --- Animate ---
ani = animation.FuncAnimation(
    fig, animate, init_func=init_plot,
    frames=int((max_samples - 50) / step_size),
    interval=20,  # speed of animation
    blit=False,
    repeat=False
)

# --- Save or show the animation ---
ani.save("./Example 1 (Fair Coin)/fair_coin.gif", writer="pillow", fps=10)
plt.show()
