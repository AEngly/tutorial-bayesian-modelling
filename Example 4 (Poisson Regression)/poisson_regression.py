# Import necessary libraries
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import CholeskyTransform
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random

# Import from sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Imports for visualization
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./bike+sharing+dataset/hour.csv")

# Add a new column with timestamps
df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
df.set_index('datetime', inplace=True)

# ############################################################################
# ####################### Data Engineering ###################################
# ############################################################################

# Set number of Fourier terms
K = 2

# Set number of samples and warmup for MCMC
num_samples = 500
num_warmup = 500
num_chains = 4

# Indicate CPU computations
numpyro.set_host_device_count(num_chains)

############## 1. One-hot encoding for categorical variables ##############

# Save a copy of the original dataframe
df_one_hot = df.copy()

# Print the number of unique categories in each categorical column
categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
print("\nCategorical columns unique value counts:\n")
for col in categorical_columns:
    unique_values = df_one_hot[col].nunique()
    print(f"Column '{col}' has {unique_values} unique categories.")

# Create a naive dataset with one-hot encoding for categorical variables
categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
continous_cols = ['temp', 'atemp', 'hum', 'windspeed']

# Set features and target variables
feature_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

# Set target variables
target_cols = ['casual', 'registered']

# Drop year columns from df to avoid redundancy
df_one_hot.drop(columns=['yr'], inplace=True)
categorical_cols.remove('yr')
feature_cols.remove('yr')

# Apply one-hot encoding to categorical variables
#encoder = OneHotEncoder(sparse_output=False, drop='first')
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_cats_one_hot = encoder.fit_transform(df_one_hot[categorical_cols])
encoded_cats_one_hot_df = pd.DataFrame(encoded_cats_one_hot, 
                              columns=encoder.get_feature_names_out(categorical_cols),
                              index=df_one_hot.index)

# Save one-hot features
one_hot_cols = encoded_cats_one_hot_df.columns.tolist()

# Define final features list
features = one_hot_cols + continous_cols

# Combine encoded categorical variables with numerical features
numerical_df = df[continous_cols]
target_df = df[target_cols]
target_df_corr = df[['cnt']]
df_one_hot = pd.concat([encoded_cats_one_hot_df, numerical_df, target_df], axis=1)
df_one_hot_corr = pd.concat([encoded_cats_one_hot_df, numerical_df, target_df_corr], axis=1)

# Save them as JAX arrays
dates = df_one_hot.index
X_one_hot = jnp.array(df_one_hot[features].values)

############## 2. Cyclical encoding for categorical variables ##############

# ----------------------
# Fourier helper function
# ----------------------
def fourier_terms(x, period, K=3, prefix='f'):
    """
    Create the first K Fourier series terms for a given time variable x.
    x should be a pd.Series.
    """
    terms = {}
    for k in range(1, K+1):
        terms[f'{prefix}_sin{k}'] = np.sin(2 * np.pi * k * x / period)
        terms[f'{prefix}_cos{k}'] = np.cos(2 * np.pi * k * x / period)
    return pd.DataFrame(terms, index=x.index)

# ----------------------
# Prepare the DataFrame
# ----------------------
df_cyclical = df.copy()
df_cyclical.index = pd.to_datetime(df_cyclical.index)

# ----------------------
# 1. Hour of day
# ----------------------
x_hour = pd.Series(df_cyclical.index.hour, index=df_cyclical.index)
period_hour = 24
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_hour, period=period_hour, K=K, prefix='hour')], axis=1)

# ----------------------
# 2. Day of week (hours)
# ----------------------
x_week = pd.Series(df_cyclical.index.weekday * 24 + df_cyclical.index.hour, index=df_cyclical.index)
period_week = 7 * 24
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_week, period=period_week, K=K, prefix='weekday')], axis=1)

# ----------------------
# 3. Day of month (hours)
# ----------------------
x_day = pd.Series((df_cyclical.index.day - 1) * 24 + df_cyclical.index.hour, index=df_cyclical.index)
period_day = 31 * 24
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_day, period=period_day, K=K, prefix='day')], axis=1)

# ----------------------
# 4. Month (hours since start of year for each month)
# ----------------------
month_start_hours = df_cyclical.index.to_period('M').to_timestamp().dayofyear - 1
x_month = pd.Series(month_start_hours * 24 + df_cyclical.index.hour, index=df_cyclical.index)
period_month = 365.25 / 12 * 24  # average month in hours
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_month, period=period_month, K=K, prefix='month')], axis=1)

# ----------------------
# 5. Season (map season to starting month)
# ----------------------
season_start_month = {1: 1, 2: 4, 3: 7, 4: 10}  # Winter, Spring, Summer, Fall
x_season = pd.Series(df_cyclical['season'].map(season_start_month), index=df_cyclical.index)
x_season_hours = pd.Series(
    ((x_season - 1) * 30.44 * 24) + df_cyclical.index.day * 24 + df_cyclical.index.hour,
    index=df_cyclical.index
)
period_season = 365.25 * 24  # full year in hours
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_season_hours, period=period_season, K=K, prefix='season')], axis=1)

# ----------------------
# 6. Full year (hours since start of year)
# ----------------------
x_year = pd.Series((df_cyclical.index.dayofyear - 1) * 24 + df_cyclical.index.hour, index=df_cyclical.index)
period_year = 365.25 * 24
df_cyclical = pd.concat([df_cyclical, fourier_terms(x_year, period=period_year, K=K, prefix='year')], axis=1)

# ----------------------
# 7. Select columns for modeling
# ----------------------

# One-hot encode weathersit
weathersit_dummies = pd.get_dummies(df_cyclical['weathersit'], prefix='weathersit')

# Combine cyclical columns, other special columns (holiday, workingday), and the new one-hot columns
cyclical_cols = [col for col in df_cyclical.columns if any(prefix in col for prefix in
                  ['hour_', 'weekday_', 'day_', 'month_', 'season_', 'year_'])]
special_cols = ['holiday', 'workingday']  # leave weathersit out, it's now one-hot
features_cyclical = cyclical_cols + special_cols + list(weathersit_dummies.columns) + continous_cols

# Merge the one-hot columns into df_cyclical
df_cyclical = pd.concat([df_cyclical, weathersit_dummies], axis=1)
df_cyclical = df_cyclical[features_cyclical].astype(float)

# Print the dataframe columns for verification
print("\nCyclical feature columns:\n", df_cyclical.columns.tolist())

# Convert to JAX array
X_cyclical = jnp.array(df_cyclical.values)

############## 3. Target variables ##############

y = jnp.array(df_one_hot[target_cols].values)
y_corr = jnp.array(df_one_hot_corr['cnt'].values)

# Description:
# Feature matrix X shape: (17379, 59)
# Target matrix y shape: (17379, 2)

# ############################################################################
# ################################# Models ###################################
# ############################################################################

def independent_nb_model(X, y=None):
    """
    Negative Binomial regression model for bike sharing data.
    Models casual and registered bike counts independently with feature-dependent overdispersion.
    - y[:, 0] = casual
    - y[:, 1] = registered
    - X: Design matrix of shape (n, d)
    - y: Target matrix of shape (n, 2)
    """
    n, d = X.shape

    # Split targets
    if y is not None:
        casual = y[:, 0]
        registered = y[:, 1]

    # Priors for casual mean
    intercept_casual = numpyro.sample("intercept_casual", dist.Normal(0, 5))
    beta_casual = numpyro.sample("beta_casual", dist.Normal(0, 3).expand([d]))
    eta_casual = intercept_casual + jnp.dot(X, beta_casual)
    mean_casual = jnp.exp(eta_casual)  # ensure positive mean

    # Priors for registered mean
    intercept_registered = numpyro.sample("intercept_registered", dist.Normal(0, 5))
    beta_registered = numpyro.sample("beta_registered", dist.Normal(0, 3).expand([d]))
    eta_registered = intercept_registered + jnp.dot(X, beta_registered)
    mean_registered = jnp.exp(eta_registered)

    # Feature-dependent overdispersion (concentration)
    intercept_concentration_casual = numpyro.sample("intercept_concentration_casual", dist.Normal(0, 3))
    beta_concentration_casual = numpyro.sample("beta_concentration_casual", dist.Normal(0, 1).expand([d]))
    eta_concentration_casual = intercept_concentration_casual + jnp.dot(X, beta_concentration_casual)
    concentration_casual = jnp.exp(eta_concentration_casual)  # >0

    intercept_concentration_registered = numpyro.sample("intercept_concentration_registered", dist.Normal(0, 3))
    beta_concentration_registered = numpyro.sample("beta_concentration_registered", dist.Normal(0, 1).expand([d]))
    eta_concentration_registered = intercept_concentration_registered + jnp.dot(X, beta_concentration_registered)
    concentration_registered = jnp.exp(eta_concentration_registered)

    # Observation model
    with numpyro.plate("data", n):
        numpyro.sample(
            "obs_casual",
            dist.NegativeBinomial2(mean=mean_casual, concentration=concentration_casual),
            obs=casual
        )

        numpyro.sample(
            "obs_registered",
            dist.NegativeBinomial2(mean=mean_registered, concentration=concentration_registered),
            obs=registered
        )

def independent_poisson_model(X, y):

    """
    Poisson regression model for bike sharing data.
    Models casual and registered bike counts independently:
    - y[:, 0] = casual
    - y[:, 1] = registered
    - X: Design matrix of shape (n, d)
    - y: Target matrix of shape (n, 2)
    """

    # Get dimensions
    n, d = X.shape  # n = number of observations, d = features

    # Split targets
    if y is not None:
        casual = y[:, 0]
        registered = y[:, 1]

    # Priors on intercepts
    intercept_casual = numpyro.sample("intercept_casual", dist.Normal(0, 5))
    intercept_registered = numpyro.sample("intercept_registered", dist.Normal(0, 5))

    # Priors on regression weights
    beta_casual = numpyro.sample("beta_casual", dist.Normal(0, 3).expand([d]))
    beta_registered = numpyro.sample("beta_registered", dist.Normal(0, 3).expand([d]))

    # Linear predictors
    eta_casual = intercept_casual + jnp.dot(X, beta_casual)
    eta_registered = intercept_registered + jnp.dot(X, beta_registered)

    # Draw parameters from Gamma priors
    # INSERT CODE HERE IF NEEDED

    # Introduce exponential transform to ensure correct supply for Poisson rates
    lambda_casual = jnp.exp(eta_casual)
    lambda_registered = jnp.exp(eta_registered)

    # ------------------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------------------
    with numpyro.plate("data", n):
        numpyro.sample("obs_casual", dist.Poisson(lambda_casual), obs=None if y is None else casual)
        numpyro.sample("obs_registered", dist.Poisson(lambda_registered), obs=None if y is None else registered)

def single_target_poisson_model(X, y):

    """
    Poisson regression model for bike sharing data.
    Models casual and registered bike counts independently:
    - y[:, 0] = casual
    - y[:, 1] = registered
    - X: Design matrix of shape (n, d)
    - y: Target matrix of shape (n, 2)
    """

    # Get dimensions
    n, d = X.shape  # n = number of observations, d = features

    # Priors on intercepts
    intercept = numpyro.sample("intercept", dist.Normal(0, 5))

    # Priors on regression weights
    beta = numpyro.sample("beta", dist.Normal(0, 3).expand([d]))

    # Linear predictors
    eta = intercept + jnp.dot(X, beta)

    # Introduce exponential transform to ensure correct supply for Poisson rates
    lambda_ = jnp.exp(eta)

    # ------------------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------------------
    with numpyro.plate("data", n):
        numpyro.sample("obs_total", dist.Poisson(lambda_), obs=None if y is None else y)

def single_target_nb_model_varying_alpha(X, y=None):
    """
    Negative Binomial regression for single-target counts with feature-dependent overdispersion.
    - X: design matrix (n, d)
    - y: target vector (n,)
    """
    n, d = X.shape

    # -------------------------
    # Priors for mean
    # -------------------------
    intercept_mean = numpyro.sample("intercept_mean", dist.Normal(0, 5))
    beta_mean = numpyro.sample("beta_mean", dist.Normal(0, 3).expand([d]))
    eta_mean = intercept_mean + jnp.dot(X, beta_mean)
    mean = jnp.exp(eta_mean)  # ensure positive mean

    # -------------------------
    # Priors for overdispersion (concentration)
    # -------------------------
    intercept_conc = numpyro.sample("intercept_conc", dist.Normal(0, 3))
    beta_conc = numpyro.sample("beta_conc", dist.Normal(0, 1).expand([d]))
    eta_conc = intercept_conc + jnp.dot(X, beta_conc)
    concentration = jnp.exp(eta_conc)  # ensure positive concentration

    # -------------------------
    # Observation model
    # -------------------------
    with numpyro.plate("data", n):
        numpyro.sample(
            "obs_total",
            dist.NegativeBinomial2(mean=mean, concentration=concentration),
            obs=y
        )

# ############################################################################
# ############################## Inference ###################################
# ############################################################################

# If cyclical features are to be used, set X = X_cyclical
X = X_cyclical  # or X_cyclical

# Split before converting to JAX arrays
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.05,      # 5% test set
    random_state=42,     # reproducibility
    shuffle=False        # shuffle the data means random sampling (must not depend on the temporal order)
)

# Convert to JAX arrays
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_test)

# Split before converting to JAX arrays
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X,
    y_corr,
    test_size=0.05,      # 5% test set
    random_state=42,     # reproducibility
    shuffle=False        # shuffle the data means random sampling (must not depend on the temporal order)
)

# Convert to JAX arrays
X_train_single = jnp.array(X_train_single)
y_train_single = jnp.array(y_train_single)
X_test_single = jnp.array(X_test_single)
y_test_single = jnp.array(y_test_single)

# Print the shapes of the X's
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")

# Set random seed
rng_key = random.PRNGKey(42)

# Sample from the posterior in single target model
nuts_kernel_single_target = NUTS(single_target_nb_model_varying_alpha)
mcmc_single_target = MCMC(nuts_kernel_single_target, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
mcmc_single_target.run(rng_key, X_train_single, y_train_single)
mcmc_single_target.print_summary()
posterior_samples_single_target = mcmc_single_target.get_samples()

# The posterior_samples dictionary contains samples for:
# - intercept_casual
# - intercept_registered
# - beta_casual
# - beta_registered

# You can now use posterior_samples for further analysis or predictions.
# Note: Further steps such as posterior predictive checks, model diagnostics,
# and evaluation on the test set can be implemented as needed.

# ############################################################################
# ############################# Prediction ###################################
# ############################################################################

# Create a Predictive object using posterior samples
predictive_corr = Predictive(single_target_nb_model_varying_alpha, posterior_samples_single_target, return_sites=["obs_total"])

# Generate posterior predictive samples for the test data
rng_key_predict = random.PRNGKey(42)

# Predict in-sample for single target (train)
predictions_in_single = predictive_corr(rng_key_predict, X_train, y=None)
y_pred_total_in_single = predictions_in_single["obs_total"]  # shape: (num_samples, n_test)

# Predict out-of-sample for single target (test)
predictions_out_single = predictive_corr(rng_key_predict, X_test, y=None)
y_pred_total_out_single = predictions_out_single["obs_total"]


# ############################################################################
# ########################## Visualization ###################################
# ############################################################################

# Get true values for test set
y_train_casual = y_train[:, 0]
y_train_registered = y_train[:, 1]
y_test_casual = y_test[:, 0]
y_test_registered = y_test[:, 1]

# Example: number of in-sample points
n_train = len(y_train_casual)  # or however many training points you have
n_test = len(y_test_casual)
n_total = n_train + n_test

# Function for prediction interval
def pred_symmetric_interval(draws, prediction_interval=0.9):

    # Compute the percentile bounds
    lower = (1 - prediction_interval) / 2 * 100
    upper = (1 + prediction_interval) / 2 * 100
    lower_bound = np.percentile(draws, lower, axis=0)
    upper_bound = np.percentile(draws, upper, axis=0)
    return lower_bound, upper_bound

# Split dates into in-sample and out-of-sample
dates_in = dates[:n_train]
dates_out = dates[n_train:]

# Compute mean and prediction intervals
def compute_summary_stats(pred_casual, pred_registered):
    prop_draws = pred_registered / (pred_registered + pred_casual)
    prop_mean = prop_draws.mean(axis=0)
    prop_lower, prop_upper = pred_symmetric_interval(prop_draws)
    total_draws = pred_registered + pred_casual
    total_mean = total_draws.mean(axis=0)
    total_lower, total_upper = pred_symmetric_interval(total_draws)
    return prop_mean, prop_lower, prop_upper, total_mean, total_lower, total_upper

# Compute mean and prediction intervals
def compute_summary_stats_single(predictions):

    # Compute the bounds
    total_mean = predictions.mean(axis=0)
    total_lower, total_upper = pred_symmetric_interval(predictions)
    return total_mean, total_lower, total_upper

# In-sample for single observation
total_mean_in_single, total_lower_in_single, total_upper_in_single = compute_summary_stats_single(y_pred_total_in_single)
print("Dimensions: ", y_pred_total_in_single.shape)

# Out-of-sample for single observation
total_mean_out_single, total_lower_out_single, total_upper_out_single = compute_summary_stats_single(y_pred_total_out_single)

# Actual totals
y_actual_total_in = y_train_casual + y_train_registered
y_actual_total_out = y_test_casual + y_test_registered

# ===== FIGURE 1: In-sample for single observation =====
plt.figure(figsize=(14, 10))
plt.plot(dates_in, total_mean_in_single, color='blue', label='Predicted Total (in-sample)')
plt.fill_between(dates_in, total_lower_in_single, total_upper_in_single, color='blue', alpha=0.2)
plt.plot(dates_in, y_train_single, color='red', alpha=0.7, label='Actual Total (in-sample)')
plt.title('Predicted Total vs Actual Total (In-sample)')
plt.xlabel('Date')
plt.ylabel('Total Count')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fig1_in_sample.png", dpi=300)
plt.show()


# ===== FIGURE 2: Out-of-sample for single observation =====
plt.figure(figsize=(14, 10))
plt.plot(dates_out, total_mean_out_single, color='green', label='Predicted Total (out-of-sample)')
plt.fill_between(dates_out, total_lower_out_single, total_upper_out_single, color='green', alpha=0.2)
plt.plot(dates_out, y_actual_total_out, color='orange', alpha=0.7, label='Actual Total (out-of-sample)')
plt.title('Predicted Total vs Actual Total (Out-of-sample)')
plt.xlabel('Date')
plt.ylabel('Total Count')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fig2_out_of_sample.png", dpi=300)
plt.show()