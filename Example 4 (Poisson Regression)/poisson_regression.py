# Import necessary libraries
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import CholeskyTransform
from numpyro.infer import MCMC, NUTS
from jax import random

# Load data
df = pd.read_csv("./bike+sharing+dataset/hour.csv")

features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

X = jnp.array(df[features].values)
casual = jnp.array(df['casual'].values)
registered = jnp.array(df['registered'].values)

# ############################################################################
# ################################# Models ###################################
# ############################################################################

def correlated_poisson_model_1(X, casual=None, registered=None):

    # Get dimensions
    n, d = X.shape  # n = number of observations, d = features

    # Priors on intercepts
    intercept_casual = numpyro.sample("intercept_casual", dist.Normal(0, 5))
    intercept_registered = numpyro.sample("intercept_registered", dist.Normal(0, 5))

    # Priors on regression weights
    beta_casual = numpyro.sample("beta_casual", dist.Normal(0, 1).expand([d]))
    beta_registered = numpyro.sample("beta_registered", dist.Normal(0, 1).expand([d]))

    # Linear predictors
    eta_casual = intercept_casual + jnp.dot(X, beta_casual)
    eta_registered = intercept_registered + jnp.dot(X, beta_registered)

    lambda_casual = jnp.exp(eta_casual)
    lambda_registered = jnp.exp(eta_registered)

    # Observation model
    with numpyro.plate("data", n):
        numpyro.sample("obs_casual", dist.Poisson(lambda_casual), obs=casual)
        numpyro.sample("obs_registered", dist.Poisson(lambda_registered), obs=registered)