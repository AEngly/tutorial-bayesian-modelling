"""
Bayesian Linear Regression for Global Temperature vs CO2
Using NumPyro for probabilistic modeling

Purpose: Knowledge-sharing session with Energinet.

Date: 22nd October 2025
Authors: Andreas Engly (ANENG) and Anders Runge Walther (AWR)

"""

# Imports for the tutorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.stats import norm
import arviz as az

# Set random seeds for reproducibility
numpyro.set_platform("cpu")


def load_data():
    """Load and prepare CO2 and temperature data"""
    print("Loading data...")

    # Load CO2 data
    co2_data = pd.read_csv('Data/co2_mm_mlo.csv', comment='#')

    # Load temperature data
    temp_data = pd.read_csv('Data/global_temperature_anomaly.csv', skiprows=1)

    # Clean temperature data - extract monthly data
    monthly_temp = []
    for _, row in temp_data.iterrows():
        year = row['Year']
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            if month in row and row[month] != '***':
                monthly_temp.append({
                    'year': year,
                    'month': month,
                    'temperature_anomaly': float(row[month])
                })

    temp_data = pd.DataFrame(monthly_temp)

    # Create month number mapping
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    temp_data['month_num'] = temp_data['month'].map(month_map)

    # Create decimal date for proper time ordering
    temp_data['decimal_date'] = temp_data['year'] + (temp_data['month_num'] - 1) / 12.0

    # Prepare CO2 data - already monthly
    co2_data['decimal_date'] = co2_data['decimal date']
    co2_data = co2_data[['decimal_date', 'average']].copy()
    co2_data.columns = ['decimal_date', 'co2_ppm']

    # Merge datasets on decimal date (with some tolerance for matching)
    data = pd.merge_asof(temp_data.sort_values('decimal_date'),
                        co2_data.sort_values('decimal_date'),
                        on='decimal_date',
                        direction='nearest',
                        tolerance=0.1)

    # Remove any remaining NaN values
    data = data.dropna()

    # Convert temperature anomaly to actual temperature
    # NASA GISTEMP uses 1951-1980 as baseline period (14.0°C)
    baseline_temp = 14.0
    data['temperature_actual'] = data['temperature_anomaly'] + baseline_temp

    # Add time since first observation (in years)
    data['time_since_start'] = data['decimal_date'] - data['decimal_date'].iloc[0]

    # Add a time of year column (day of year / total days in year)
    # Convert decimal date to day of year
    data['day_of_year'] = ((data['decimal_date'] - data['year']) * 365.25).round().astype(int)

    # Calculate total days in each year (accounting for leap years)
    data['is_leap_year'] = ((data['year'] % 4 == 0) & (data['year'] % 100 != 0)) | (data['year'] % 400 == 0)
    data['days_in_year'] = data['is_leap_year'].map({True: 366, False: 365})

    # Time of year as fraction (0 to 1)
    data['time_of_year'] = data['day_of_year'] / data['days_in_year']

    # Clean up intermediate columns
    data = data.drop(['day_of_year', 'is_leap_year', 'days_in_year'], axis=1)

    # Return the data
    return data

# Load the data
data = load_data()

# Specify the regression model to be used
def bayesian_linear_regression(X, y=None):
    """
    Bayesian temperature regression model.

    Model: T = β₀ + β₁ * log₂(CO₂/CO₂₀) + ε
    where T is actual temperature (°C)
    and CO₂₀ is the baseline CO₂ concentration (315 ppm)
    """

    # Step 1: Set hyperparameters
    # β₀: temperature at baseline CO₂, β₁: temperature sensitivity to log₂(CO₂)
    beta_prior_mean = jnp.array([14.0, 3.0])  # [temp at baseline CO₂, sensitivity]
    beta_prior_std = jnp.array([5.0, 1.5])    # [temp at baseline CO₂, sensitivity]
    sigma_prior_scale = 1.0

    # Step 2a: Define priors for regression coefficients
    beta = numpyro.sample("beta", dist.Normal(beta_prior_mean, beta_prior_std))

    # Step 2b: Define prior for noise standard deviation
    sigma = numpyro.sample("sigma", dist.Exponential(sigma_prior_scale))

    # Step 3: Define linear relationship using matrix-vector product
    mu = X @ beta

    # Step 4: Define likelihood
    numpyro.sample("temperature", dist.Normal(mu, sigma), obs=y)

# Specify the regression model to be used
def bayesian_linear_regression_heteroscedastic(X, y=None):
    """
    Bayesian temperature regression model with heteroscedasticity.

    Model: T = β₀ + β₁ * log₂(CO₂/CO₂₀) + ε
    where T is actual temperature (°C)
    and CO₂₀ is the baseline CO₂ concentration (315 ppm)
    and ε is heteroscedastic noise.
    """

    # Step 1: Set hyperparameters
    # β₀: temperature at baseline CO₂, β₁: temperature sensitivity to log₂(CO₂)
    beta_prior_mean = jnp.array([14.0, 3.0])  # [temp at baseline CO₂, sensitivity]
    beta_prior_std = jnp.array([5.0, 1.5])    # [temp at baseline CO₂, sensitivity]
    gamma_prior_mean = jnp.array([1.0, 2.0])  # [noise scaling factor]
    gamma_prior_std = jnp.array([0.5, 0.5])  # [noise scaling factor]

    # Step 2a: Define priors for regression coefficients
    beta = numpyro.sample("beta", dist.Normal(beta_prior_mean, beta_prior_std))

    # Step 2b: Define prior for noise standard deviation
    gamma = numpyro.sample("gamma", dist.Normal(gamma_prior_mean, gamma_prior_std))

    # Step 3: Define linear relationship using matrix-vector product
    mu = X @ beta
    sigma = jnp.exp(X @ gamma)

    # Step 4: Define likelihood
    numpyro.sample("temperature", dist.Normal(mu, sigma), obs=y)

def prepare_design_matrix(data):
    """
    Prepare design matrix X for regression with logarithmic CO2 relationship.
    X = [1, log2(CO2/CO2_baseline)] where 1 is the intercept column

    This implements the climate sensitivity equation:
    ΔT = S * log2(CO2/CO2_baseline)
    """
    # Calculate log2(CO2) for climate sensitivity
    co2_ratio = jnp.array(data['co2_ppm'].values)
    log2_co2 = jnp.log2(co2_ratio/co2_ratio[0])

    # Create design matrix: [intercept, log2(CO2)]
    X = jnp.column_stack([
        jnp.ones(len(data)),  # intercept column (all 1s)
        log2_co2  # log2(CO2) column
    ])
    return X

def run_mcmc(data, model=None, num_samples=5000, num_warmup=500, num_chains=4):

    """
    Run MCMC sampling for the Bayesian regression model with multiple chains.
    """

    # Make sure the chains are used in the MCMC by enabling parallelization
    numpyro.set_host_device_count(num_chains)

    # Prepare design matrix and target
    X = prepare_design_matrix(data)
    y = jnp.array(data['temperature_actual'].values)

    # Use provided model or default to bayesian_linear_regression
    if model is None:
        model = bayesian_linear_regression

    # Create NUTS kernel
    kernel = NUTS(model)

    # Run MCMC with multiple chains
    print(f"Running MCMC with {num_chains} chains...")
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)
    mcmc.run(random.PRNGKey(42), X, y)

    # Return the MCMC samples
    return mcmc

# Get temperature and CO2
dates = data['decimal_date']
temperature_index = (data['temperature_actual']/data["temperature_actual"].iloc[0]) * 100
co2_ppm_index = (data['co2_ppm']/data["co2_ppm"].iloc[0]) * 100
temperature = data['temperature_actual']
co2_ppm = data['co2_ppm']

# Plot 1: Plot CO2 and temperature data
plt.figure(figsize=(10, 6))
plt.plot(dates, co2_ppm_index, label='CO2')
plt.plot(dates, temperature_index, label='Temperature')
plt.title('CO2 and Temperature Index (100 = 1958)', pad=20)
plt.xlabel('Year')
plt.ylabel('Index (100 = 1958)')
plt.legend()
plt.grid(True)
plt.savefig('co2_temperature_index.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Plot CO2 on first axis and temperature on second axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis: CO2
ax1.plot(dates, co2_ppm, 'b-', label='CO2', linewidth=2)
ax1.set_xlabel('Year')
ax1.set_ylabel('CO2 (ppm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

# Second y-axis: Temperature
ax2 = ax1.twinx()
ax2.plot(dates, temperature, 'r-', label='Temperature', linewidth=2)
ax2.set_ylabel('Temperature (°C)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('CO2 and Temperature', pad=20)
plt.tight_layout()
plt.savefig('co2_temperature_twinned.png', dpi=300, bbox_inches='tight')
plt.show()

# Now make a single plot of rolling window mean temperature vs rolling window variance of temperature anomalies
rolling_window_size = 5*12 # 5 years

# Convert pandas Series to JAX array
temperature_array = jnp.array(temperature.values)

# Calculate rolling window mean and variance in a single loop
rolling_window_mean = []
rolling_window_variance = []
for i in range(len(temperature_array)):
    if i < rolling_window_size - 1:
        rolling_window_mean.append(jnp.nan)
        rolling_window_variance.append(jnp.nan)
    else:
        rolling_window_mean.append(jnp.mean(temperature_array[i-rolling_window_size+1:i+1]))
        rolling_window_variance.append(jnp.var(temperature_array[i-rolling_window_size+1:i+1]))

# Plot them against each other
plt.figure(figsize=(12, 6))
plt.plot(rolling_window_mean, rolling_window_variance, 'bo')
plt.xlabel('Rolling Window Mean Temperature (°C)')
plt.ylabel('Rolling Window Variance of Temperature (°C²)')
plt.title('Rolling Window Mean Temperature vs Rolling Window Variance of Temperature', pad=20)
plt.savefig('rolling_window_mean_temperature_vs_rolling_window_variance_of_temperature.png', dpi=300, bbox_inches='tight')
plt.show()

# Run the analysis
mcmc = run_mcmc(data)
samples = mcmc.get_samples()

# Print summary statistics
print("\nTemperature vs CO₂ Analysis:")
print(f"β₀ (temperature at 315 ppm CO₂): {jnp.mean(samples['beta'][:, 0]):.3f} ± {jnp.std(samples['beta'][:, 0]):.3f} °C")
print(f"β₁ (temperature sensitivity): {jnp.mean(samples['beta'][:, 1]):.3f} ± {jnp.std(samples['beta'][:, 1]):.3f} °C per log₂(CO₂/315)")
print(f"σ (noise): {jnp.mean(samples['sigma']):.3f} ± {jnp.std(samples['sigma']):.3f} °C")

# Check convergence diagnostics
print(f"\nConvergence Diagnostics:")
print(f"Number of chains: {mcmc.num_chains}")
print(f"Number of samples per chain: {mcmc.num_samples}")
print(f"Warmup samples per chain: {mcmc.num_warmup}")
print(f"Total samples: {mcmc.num_samples * mcmc.num_chains}")

# Check effective sample size (if available)
try:
    ess = mcmc.get_extra_fields()['diverging']
    print(f"Divergent transitions: {jnp.sum(ess)}")
except:
    print("Divergence info not available")

# Can we make a diagnostic plot of the chains? Or any other relevant diagnostics?
# We can use the arviz library to plot the chains
az.plot_trace(mcmc)
plt.savefig('mcmc_chains.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot posterior samples side by side with prior distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot β₀ (intercept) - Temperature at baseline CO₂
ax1.hist(samples['beta'][:, 0], bins=50, density=True, alpha=0.7,
         label='Posterior', color='blue', edgecolor='black')

# Add credibility intervals for β₀
beta0_map = jnp.mean(samples['beta'][:, 0])
beta0_ci_50 = jnp.percentile(samples['beta'][:, 0], jnp.array([25, 75]))
beta0_ci_95 = jnp.percentile(samples['beta'][:, 0], jnp.array([2.5, 97.5]))

ax1.axvline(beta0_map, color='blue', linestyle='-', linewidth=2, alpha=0.8, label=f'Maximum a Posteriori (MAP): {beta0_map:.2f}')
ax1.axvline(beta0_ci_50[0], color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'50% Credibility Interval: [{beta0_ci_50[0]:.2f}, {beta0_ci_50[1]:.2f}]')
ax1.axvline(beta0_ci_50[1], color='blue', linestyle=':', linewidth=2, alpha=0.6)
ax1.axvline(beta0_ci_95[0], color='blue', linestyle='-.', linewidth=1.5, alpha=0.4, label=f'95% Credibility Interval: [{beta0_ci_95[0]:.2f}, {beta0_ci_95[1]:.2f}]')
ax1.axvline(beta0_ci_95[1], color='blue', linestyle='-.', linewidth=1.5, alpha=0.4)

ax1.set_xlabel('β₀ (Temperature at baseline CO₂)')
ax1.set_ylabel('Density')
ax1.set_title('Posterior: β₀', pad=20)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot β₁ (slope) - Temperature sensitivity to log₂(CO₂)
ax2.hist(samples['beta'][:, 1], bins=50, density=True, alpha=0.7,
         label='Posterior', color='green', edgecolor='black')

# Add credibility intervals for β₁
beta1_map = jnp.mean(samples['beta'][:, 1])
beta1_ci_50 = jnp.percentile(samples['beta'][:, 1], jnp.array([25, 75]))
beta1_ci_95 = jnp.percentile(samples['beta'][:, 1], jnp.array([2.5, 97.5]))

ax2.axvline(beta1_map, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Maximum a Posteriori (MAP): {beta1_map:.2f}')
ax2.axvline(beta1_ci_50[0], color='green', linestyle=':', linewidth=2, alpha=0.6, label=f'50% Credibility Interval: [{beta1_ci_50[0]:.2f}, {beta1_ci_50[1]:.2f}]')
ax2.axvline(beta1_ci_50[1], color='green', linestyle=':', linewidth=2, alpha=0.6)
ax2.axvline(beta1_ci_95[0], color='green', linestyle='-.', linewidth=1.5, alpha=0.4, label=f'95% Credibility Interval: [{beta1_ci_95[0]:.2f}, {beta1_ci_95[1]:.2f}]')
ax2.axvline(beta1_ci_95[1], color='green', linestyle='-.', linewidth=1.5, alpha=0.4)

ax2.set_xlabel('β₁ (Temperature sensitivity to log₂(CO₂))')
ax2.set_ylabel('Density')
ax2.set_title('Posterior: β₁', pad=20)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot σ (noise standard deviation)
ax3.hist(samples['sigma'], bins=50, density=True, alpha=0.7,
         label='Posterior', color='red', edgecolor='black')

# Add credibility intervals for σ
sigma_mean = jnp.mean(samples['sigma'])
sigma_ci_50 = jnp.percentile(samples['sigma'], jnp.array([25, 75]))
sigma_ci_95 = jnp.percentile(samples['sigma'], jnp.array([2.5, 97.5]))

ax3.axvline(sigma_mean, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {sigma_mean:.3f}')
ax3.axvline(sigma_ci_50[0], color='red', linestyle=':', linewidth=2, alpha=0.6, label=f'50% CI: [{sigma_ci_50[0]:.3f}, {sigma_ci_50[1]:.3f}]')
ax3.axvline(sigma_ci_50[1], color='red', linestyle=':', linewidth=2, alpha=0.6)
ax3.axvline(sigma_ci_95[0], color='red', linestyle='-.', linewidth=1.5, alpha=0.4, label=f'95% CI: [{sigma_ci_95[0]:.3f}, {sigma_ci_95[1]:.3f}]')
ax3.axvline(sigma_ci_95[1], color='red', linestyle='-.', linewidth=1.5, alpha=0.4)

ax3.set_xlabel('σ (Noise Standard Deviation)')
ax3.set_ylabel('Density')
ax3.set_title('Posterior: σ', pad=20)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('posterior_samples_regression_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate posterior predictive samples
X = prepare_design_matrix(data)
y = jnp.array(data['temperature_actual'].values)
predictive = Predictive(bayesian_linear_regression, samples)
predictions = predictive(random.PRNGKey(42), X)['temperature']

# Then plot the predictions from the MCMC model (time vs. temperature)
# Show P5 to P95 confidence intervals
plt.fill_between(dates, np.percentile(predictions, 5, axis=0), np.percentile(predictions, 95, axis=0), alpha=0.5, label='5-95% Prediction Interval')
plt.plot(dates, np.mean(predictions, axis=0), 'b-', label='Mean', linewidth=2)
plt.plot(dates, temperature, 'r--', alpha=0.7, label='Observations')
plt.legend()
plt.title('Bayesian Linear Regression (Global Surface Temperature)', pad=20)
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.savefig('bayesian_regression_fit.png', dpi=300, bbox_inches='tight')
plt.show()


# Fit a deterministic model to CO2 to make a scenario
# f(t) = exp(beta0 + beta1 * t + beta2 * sine(2 * pi * t / 12) + beta3 * cosine(2 * pi * t / 12))
# where t is the time in years since 1958

# Fit it using the normal equations
# Step 1: Create the design matrix
X = jnp.column_stack([
    jnp.ones(len(data)),
    jnp.array(data['time_since_start'].values),
    jnp.array(data['time_since_start'].values)**2,
    jnp.sin(2 * jnp.pi * data['time_of_year'].values),  # Annual cycle
    jnp.cos(2 * jnp.pi * data['time_of_year'].values)   # Annual cycle
])
# Step 2: Solve for the coefficients (log-linear model for exponential growth)
beta = jnp.linalg.inv(X.T @ X) @ X.T @ jnp.log(jnp.array(data['co2_ppm'].values))

# Step 3: Generate CO2 model predictions (exponential)
co2_model = jnp.exp(X @ beta)

# Plot CO2 model on first axis and temperature on second axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis: CO2 model
ax1.plot(dates, co2_model, 'b-', label='CO2 Model', linewidth=2)
ax1.plot(dates, data['co2_ppm'], 'b--', alpha=0.7, label='CO2 Data')
ax1.set_xlabel('Year')
ax1.set_ylabel('CO2 (ppm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

# Second y-axis: Temperature
ax2 = ax1.twinx()
ax2.plot(dates, data['temperature_actual'], 'r-', label='Temperature', linewidth=2)
ax2.set_ylabel('Temperature (°C)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('CO2 Model vs Temperature Data', pad=20)
plt.tight_layout()
plt.savefig('co2_model_vs_data.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot CO2 model residuals
co2_residuals = co2_model - data['co2_ppm'].values
plt.figure(figsize=(12, 6))
plt.plot(dates, co2_residuals, 'b-', linewidth=1, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Perfect Fit')
plt.fill_between(dates, -5, 5, alpha=0.2, color='gray', label='±5 ppm')
plt.xlabel('Year')
plt.ylabel('CO₂ Residuals (Model - Data) [ppm]')
plt.title('CO₂ Model Residuals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('co2_model_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# Print residual statistics
print(f"\nCO₂ Model Residual Statistics:")
print(f"Mean residual: {jnp.mean(co2_residuals):.3f} ppm")
print(f"Std residual: {jnp.std(co2_residuals):.3f} ppm")
print(f"RMSE: {jnp.sqrt(jnp.mean(co2_residuals**2)):.3f} ppm")
print(f"Max absolute residual: {jnp.max(jnp.abs(co2_residuals)):.3f} ppm")

# Forecast temperature until 2100 using the deterministic CO2 model
print("Generating temperature forecast to 2100...")

# Create future dates (monthly from 2025 to 2100)
future_years = jnp.arange(2025, 2101)
future_months = jnp.arange(1, 13)
future_dates = []
future_co2 = []

for year in future_years:
    for month in future_months:
        # Calculate decimal date
        decimal_date = year + (month - 1) / 12.0
        future_dates.append(decimal_date)

        # Calculate time since start (2025 - 1958 = 67 years)
        time_since_start = decimal_date - 1958.0
        time_of_year = (month - 1) / 12.0

        # Create design matrix for CO2 model
        X_future = jnp.array([1.0, time_since_start, time_since_start**2,
                            jnp.sin(2 * jnp.pi * time_of_year),
                            jnp.cos(2 * jnp.pi * time_of_year)])

        # Predict CO2 using the deterministic model
        log_co2 = X_future @ beta
        co2_ppm = jnp.exp(log_co2)
        future_co2.append(co2_ppm)

future_dates = jnp.array(future_dates)
future_co2 = jnp.array(future_co2)

# Calculate log2(CO2/CO2_baseline) for temperature prediction
# Why use 315.0 as the baseline?
# Because it is the average CO2 concentration for the 1950s and 1960s.
# This is a good proxy for the baseline temperature.
# We can also use the average CO2 concentration for the 1950s and 1960s as the baseline.
# But this is 1958 because this is the year of the first WMO measurements.
# The Lannternists conferences concluded in 1958 that the

# Would that not just be captured in the intercept of the Bayesian regression?
# Yes, butthat now depends on assumptions and choice of climte sensitivity we

future_log2_co2 = jnp.log2(future_co2/data['co2_ppm'].iloc[0])

# Create design matrix for temperature prediction
future_X = jnp.column_stack([
    jnp.ones(len(future_dates)),
    future_log2_co2
])

# Generate temperature predictions using NumPyro Predictive
future_predictive = Predictive(bayesian_linear_regression, samples)
future_predictions = future_predictive(random.PRNGKey(42), future_X)['temperature']

# Plot the forecast
plt.figure(figsize=(15, 10))

# Subplot 1: CO2 forecast
plt.subplot(2, 1, 1)
plt.plot(dates, data['co2_ppm'], 'b-', label='Historical CO₂ (ppm)', linewidth=2)
plt.plot(future_dates, future_co2, 'r-', label='Extrapolated CO₂ (ppm)', linewidth=2)
plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('CO₂ (ppm)')
plt.title('Atmospheric CO₂ Concentration: Observed and Extrapolated')
plt.legend()
plt.grid(True, alpha=0.3)

# Convert predictions to temperature anomalies (subtract baseline)
baseline_temp = 14.0  # Same baseline as used in data loading
historical_anomaly = temperature - baseline_temp
future_anomaly_predictions = future_predictions - baseline_temp

# Subplot 2: Temperature anomaly forecast
plt.subplot(2, 1, 2)
plt.plot(dates, historical_anomaly, 'b-', label='Historical Temperature Anomaly', linewidth=2)
plt.fill_between(dates,
                jnp.percentile(predictions - baseline_temp, 5, axis=0),
                jnp.percentile(predictions - baseline_temp, 95, axis=0),
                alpha=0.2, color='blue', label='Historical 5-95% Prediction Interval')
plt.fill_between(future_dates,
                jnp.percentile(future_anomaly_predictions, 5, axis=0),
                jnp.percentile(future_anomaly_predictions, 95, axis=0),
                alpha=0.3, color='red', label='Forecast 5-95% Prediction Interval')
plt.plot(future_dates, jnp.mean(future_anomaly_predictions, axis=0), 'r-', label='Temperature Anomaly Forecast', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Baseline (1951-1980)')
plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Anomaly: Historical and Forecast')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_forecast_2100.png', dpi=300, bbox_inches='tight')
plt.show()

# Print forecast summary
print(f"\nTemperature Anomaly Forecast Summary:")
print(f"2100 CO₂: {future_co2[-1]:.1f} ppm")
print(f"2100 Temperature Anomaly: {jnp.mean(future_anomaly_predictions[:, -1]):.2f} °C")
print(f"2100 Temperature Anomaly Range: {jnp.percentile(future_anomaly_predictions[:, -1], 5):.2f} - {jnp.percentile(future_anomaly_predictions[:, -1], 95):.2f} °C")
print(f"Temperature anomaly increase from 2025: {jnp.mean(future_anomaly_predictions[:, -1]) - jnp.mean(future_anomaly_predictions[:, 0]):.2f} °C")
print(f"Total warming from 1958 baseline: {jnp.mean(future_anomaly_predictions[:, -1]):.2f} °C")

# Run the Bayesian linear regression with heteroscedasticity and plot the posteriors for the gamma parameters
mcmc = run_mcmc(data, model=bayesian_linear_regression_heteroscedastic)
samples = mcmc.get_samples()

# Plot the posteriors for the gamma parameters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot γ₁ (noise scaling factor)
ax1.hist(samples['gamma'][:, 0], bins=50, density=True, alpha=0.7,
         label='Posterior', color='blue', edgecolor='black')

# Add credibility intervals for γ₀ (noise scaling factor)
gamma0_map = jnp.mean(samples['gamma'][:, 0])
gamma0_ci_50 = jnp.percentile(samples['gamma'][:, 0], jnp.array([25, 75]))
gamma0_ci_95 = jnp.percentile(samples['gamma'][:, 0], jnp.array([2.5, 97.5]))

ax1.axvline(gamma0_map, color='blue', linestyle='-', linewidth=2, alpha=0.8, label=f'Maximum a Posteriori (MAP): {gamma0_map:.2f}')
ax1.axvline(gamma0_ci_50[0], color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'50% Credibility Interval: [{gamma0_ci_50[0]:.2f}, {gamma0_ci_50[1]:.2f}]')
ax1.axvline(gamma0_ci_50[1], color='blue', linestyle=':', linewidth=2, alpha=0.6)
ax1.axvline(gamma0_ci_95[0], color='blue', linestyle='-.', linewidth=1.5, alpha=0.4, label=f'95% Credibility Interval: [{gamma0_ci_95[0]:.2f}, {gamma0_ci_95[1]:.2f}]')
ax1.axvline(gamma0_ci_95[1], color='blue', linestyle='-.', linewidth=1.5, alpha=0.4)

ax1.set_xlabel('γ₀ (Noise scaling factor)')
ax1.set_ylabel('Density')
ax1.set_title('Posterior: γ₀ (Noise scaling factor)', pad=20)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot γ₂ (noise scaling factor)
ax2.hist(samples['gamma'][:, 1], bins=50, density=True, alpha=0.7,
         label='Posterior', color='green', edgecolor='black')

# Add credibility intervals for γ₁ (noise scaling factor)
gamma1_map = jnp.mean(samples['gamma'][:, 1])
gamma1_ci_50 = jnp.percentile(samples['gamma'][:, 1], jnp.array([25, 75]))
gamma1_ci_95 = jnp.percentile(samples['gamma'][:, 1], jnp.array([2.5, 97.5]))

ax2.axvline(gamma1_map, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Maximum a Posteriori (MAP): {gamma1_map:.2f}')
ax2.axvline(gamma1_ci_50[0], color='green', linestyle=':', linewidth=2, alpha=0.6, label=f'50% Credibility Interval: [{gamma1_ci_50[0]:.2f}, {gamma1_ci_50[1]:.2f}]')
ax2.axvline(gamma1_ci_50[1], color='green', linestyle=':', linewidth=2, alpha=0.6)
ax2.axvline(gamma1_ci_95[0], color='green', linestyle='-.', linewidth=1.5, alpha=0.4, label=f'95% Credibility Interval: [{gamma1_ci_95[0]:.2f}, {gamma1_ci_95[1]:.2f}]')
ax2.axvline(gamma1_ci_95[1], color='green', linestyle='-.', linewidth=1.5, alpha=0.4)

ax2.set_xlabel('γ₁ (Noise scaling factor)')
ax2.set_ylabel('Density')
ax2.set_title('Posterior: γ₁ (Noise scaling factor)', pad=20)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig('posterior_samples_gamma_parameters.png', dpi=300, bbox_inches='tight')
plt.show()

# In the same plot, plot the heteroscedasticity parameter against the homoscedasticity model for forecasting to 2100 (color the prediction intervals differently)

# Run homoscedastic model for comparison
print("Running homoscedastic model for comparison...")
mcmc_homo = run_mcmc(data, model=bayesian_linear_regression)
samples_homo = mcmc_homo.get_samples()

# Generate future predictions for both models
print("Generating future predictions for both models...")

# Future dates and CO2 (same as before)
future_log2_co2 = jnp.log2(future_co2/data['co2_ppm'].iloc[0])

# Create design matrix for temperature prediction
future_X = jnp.column_stack([
    jnp.ones(len(future_dates)),
    future_log2_co2
])

# Prepare future design matrix for temperature prediction
future_log2_co2 = jnp.log2(future_co2 / data['co2_ppm'].iloc[0])
X_future = jnp.column_stack([jnp.ones(len(future_log2_co2)), future_log2_co2])

# Generate predictions for heteroscedastic model
predictive_hetero = Predictive(bayesian_linear_regression_heteroscedastic, samples)
future_predictions_hetero = predictive_hetero(random.PRNGKey(123), X_future)['temperature']

# Generate predictions for homoscedastic model
predictive_homo = Predictive(bayesian_linear_regression, samples_homo)
future_predictions_homo = predictive_homo(random.PRNGKey(123), X_future)['temperature']

# Convert to temperature anomalies (subtract baseline)
baseline_temp = 14.0
future_anomaly_hetero = future_predictions_hetero - baseline_temp
future_anomaly_homo = future_predictions_homo - baseline_temp

# Calculate prediction intervals
hetero_mean = jnp.mean(future_anomaly_hetero, axis=0)
hetero_ci_5 = jnp.percentile(future_anomaly_hetero, 5, axis=0)
hetero_ci_95 = jnp.percentile(future_anomaly_hetero, 95, axis=0)

homo_mean = jnp.mean(future_anomaly_homo, axis=0)
homo_ci_5 = jnp.percentile(future_anomaly_homo, 5, axis=0)
homo_ci_95 = jnp.percentile(future_anomaly_homo, 95, axis=0)

# Create comparison plot
plt.figure(figsize=(15, 10))

# Subplot 1: CO2 forecast
plt.subplot(2, 1, 1)
plt.plot(dates, data['co2_ppm'], 'b-', label='Historical CO₂ (ppm)', linewidth=2)
plt.plot(future_dates, future_co2, 'r-', label='Extrapolated CO₂ (ppm)', linewidth=2)
plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('CO₂ (ppm)')
plt.title('Atmospheric CO₂ Concentration: Observed and Extrapolated')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Temperature anomaly forecast comparison
plt.subplot(2, 1, 2)

# Historical data
historical_anomaly = data['temperature_actual'] - baseline_temp
plt.plot(dates, historical_anomaly, 'k-', label='Historical Temperature Anomaly', linewidth=2, alpha=0.8)

# Heteroscedastic model predictions
plt.plot(future_dates, hetero_mean, 'r-', label='Heteroscedastic Model (Mean)', linewidth=2)
plt.fill_between(future_dates, hetero_ci_5, hetero_ci_95, alpha=0.3, color='red',
                label='Heteroscedastic Model (5-95% CI)')

# Homoscedastic model predictions
plt.plot(future_dates, homo_mean, 'b-', label='Homoscedastic Model (Mean)', linewidth=2)
plt.fill_between(future_dates, homo_ci_5, homo_ci_95, alpha=0.2, color='blue',
                label='Homoscedastic Model (5-95% CI)')

plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Anomaly Forecast: Heteroscedastic vs Homoscedastic Models')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_forecast_comparison_hetero_vs_homo.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comparison summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"2100 CO₂: {future_co2[-1]:.1f} ppm")
print(f"\nHeteroscedastic Model:")
print(f"  2100 Temperature Anomaly: {hetero_mean[-1]:.2f} °C")
print(f"  2100 Temperature Anomaly Range: {hetero_ci_5[-1]:.2f} - {hetero_ci_95[-1]:.2f} °C")
print(f"  Uncertainty Width: {hetero_ci_95[-1] - hetero_ci_5[-1]:.2f} °C")

print(f"\nHomoscedastic Model:")
print(f"  2100 Temperature Anomaly: {homo_mean[-1]:.2f} °C")
print(f"  2100 Temperature Anomaly Range: {homo_ci_5[-1]:.2f} - {homo_ci_95[-1]:.2f} °C")
print(f"  Uncertainty Width: {homo_ci_95[-1] - homo_ci_5[-1]:.2f} °C")

print(f"\nDifference (Hetero - Homo):")
print(f"  Mean difference: {hetero_mean[-1] - homo_mean[-1]:.2f} °C")
print(f"  Uncertainty difference: {(hetero_ci_95[-1] - hetero_ci_5[-1]) - (homo_ci_95[-1] - homo_ci_5[-1]):.2f} °C")

# =============================================================================
# OUT-OF-SAMPLE MODEL EVALUATION
# =============================================================================
# This section evaluates model performance on unseen data using:
# 1. Train-test split (80% training, 20% testing)
# 2. CRPS (Continuous Ranked Probability Score) - measures probabilistic accuracy
# 3. WAIC (Watanabe-Akaike Information Criterion) - balances fit vs complexity

print("\n" + "="*60)
print("OUT-OF-SAMPLE MODEL EVALUATION")
print("="*60)

# Step 1: Split data into training and testing sets
print("Splitting data into training (80%) and testing (20%) sets...")
train_data = data.sample(frac=0.8, random_state=42)  # 80% for training
test_data = data.drop(train_data.index)              # 20% for testing

print(f"Training data: {len(train_data)} observations")
print(f"Testing data: {len(test_data)} observations")
print(f"Training fraction: {len(train_data)/len(data)*100:.1f}%")

# Step 2: Fit homoscedastic model on training data
print("\nFitting homoscedastic model on training data...")
mcmc_train_homo = run_mcmc(train_data, model=bayesian_linear_regression, num_samples=1000)
samples_train_homo = mcmc_train_homo.get_samples()

# Step 3: Fit heteroscedastic model on training data
print("Fitting heteroscedastic model on training data...")
mcmc_train_hetero = run_mcmc(train_data, model=bayesian_linear_regression_heteroscedastic, num_samples=1000)
samples_train_hetero = mcmc_train_hetero.get_samples()

# Step 4: Prepare test data for prediction
print("Preparing test data for prediction...")
X_test = prepare_design_matrix(test_data)
y_test = jnp.array(test_data['temperature_actual'].values)

# Step 5: Generate predictions on test data
print("Generating predictions on test data...")

# Homoscedastic model predictions
predictive_test_homo = Predictive(bayesian_linear_regression, samples_train_homo)
predictions_test_homo = predictive_test_homo(random.PRNGKey(123), X_test)['temperature']

# Heteroscedastic model predictions
predictive_test_hetero = Predictive(bayesian_linear_regression_heteroscedastic, samples_train_hetero)
predictions_test_hetero = predictive_test_hetero(random.PRNGKey(123), X_test)['temperature']

# =============================================================================
# CRPS CALCULATION
# =============================================================================
# CRPS (Continuous Ranked Probability Score) measures the quality of probabilistic forecasts
# Lower CRPS values indicate better predictions
# Formula: CRPS = E|F - 1_{y ≤ x}| - 0.5 * E|F' - F''|
# where F is the forecast CDF and y is the observation

def crps_ensemble(observation, forecast_samples):
    """
    Compute CRPS for a single observation given predictive samples.

    The CRPS measures the difference between the forecast CDF and the observation.
    It combines both reliability (calibration) and resolution (sharpness).

    Parameters
    ----------
    observation : float
        The observed value.
    forecast_samples : array-like
        Predictive samples (posterior draws or ensemble forecasts).

    Returns
    -------
    crps : float
        The CRPS score (lower is better).
    """
    forecast_samples = np.sort(forecast_samples)

    # Term 1: Mean absolute error between forecast and observation
    # This measures how close the forecast is to the observation
    term1 = np.mean(np.abs(forecast_samples - observation))

    # Term 2: Half the mean absolute differences between forecast samples
    # This measures the internal consistency of the forecast ensemble
    term2 = 0.5 * np.mean(np.abs(forecast_samples[:, None] - forecast_samples[None, :]))

    return term1 - term2

def crps_vectorized(observations, predictions):
    """
    Efficient CRPS calculation for all observations at once.

    This function calculates CRPS for multiple observations simultaneously
    while being memory-efficient to avoid creating massive arrays.

    Parameters
    ----------
    observations : array-like
        Observed values for all test points.
    predictions : array-like, shape (n_samples, n_obs)
        Predictive samples for all test points.

    Returns
    -------
    crps_scores : array
        CRPS scores for each observation.
    """
    n_obs = len(observations)
    n_samples = predictions.shape[0]

    # Sort predictions for each observation
    predictions_sorted = np.sort(predictions, axis=0)

    # Term 1: Mean absolute error between forecast and observation
    # Vectorized calculation for all observations at once
    term1 = np.mean(np.abs(predictions_sorted - observations[None, :]), axis=0)

    # Term 2: Half the mean absolute differences between forecast samples
    # Calculate for each observation separately to avoid memory explosion
    term2 = np.zeros(n_obs)
    for i in range(n_obs):
        pred_i = predictions_sorted[:, i]
        # Calculate pairwise differences for this observation only
        diff_matrix = np.abs(pred_i[:, None] - pred_i[None, :])
        term2[i] = 0.5 * np.mean(diff_matrix)

    return term1 - term2

# Calculate CRPS scores for both models
print("Calculating CRPS scores for both models...")
crps_scores_homo = crps_vectorized(y_test, predictions_test_homo)
crps_scores_hetero = crps_vectorized(y_test, predictions_test_hetero)

# Calculate mean CRPS scores
mean_crps_homo = np.mean(crps_scores_homo)
mean_crps_hetero = np.mean(crps_scores_hetero)

# =============================================================================
# RESULTS VISUALIZATION
# =============================================================================
# Create comprehensive visualization of model comparison results

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(f"CRPS Scores (lower is better):")
print(f"  Homoscedastic Model: {mean_crps_homo:.4f} °C")
print(f"  Heteroscedastic Model: {mean_crps_hetero:.4f} °C")
print(f"  Difference (Homo - Hetero): {mean_crps_homo - mean_crps_hetero:.4f} °C")

# Determine which model performs better
if mean_crps_homo < mean_crps_hetero:
    print(f"\nCRPS Winner: Homoscedastic Model (by {mean_crps_hetero - mean_crps_homo:.4f} °C)")
else:
    print(f"\nCRPS Winner: Heteroscedastic Model (by {mean_crps_homo - mean_crps_hetero:.4f} °C)")

# Create CRPS comparison plot
print("\nCreating CRPS comparison visualization...")
plt.figure(figsize=(12, 8))

# Plot 1: CRPS scores distribution
plt.subplot(2, 2, 1)
plt.hist(crps_scores_homo, bins=20, alpha=0.7, label='Homoscedastic', color='blue', density=True)
plt.hist(crps_scores_hetero, bins=20, alpha=0.7, label='Heteroscedastic', color='red', density=True)
plt.axvline(mean_crps_homo, color='blue', linestyle='-', linewidth=2, label=f'Mean CRPS: {mean_crps_homo:.4f}')
plt.axvline(mean_crps_hetero, color='red', linestyle='-', linewidth=2, label=f'Mean CRPS: {mean_crps_hetero:.4f}')
plt.xlabel('CRPS Score (°C)')
plt.ylabel('Density')
plt.title('CRPS Scores Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test predictions vs actual
plt.subplot(2, 2, 2)
plt.scatter(y_test, np.mean(predictions_test_homo, axis=0), alpha=0.6, label='Homoscedastic', color='blue')
plt.scatter(y_test, np.mean(predictions_test_hetero, axis=0), alpha=0.6, label='Heteroscedastic', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5, label='Perfect Prediction')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Prediction intervals coverage
plt.subplot(2, 2, 3)
homo_ci_5 = np.percentile(predictions_test_homo, 5, axis=0)
homo_ci_95 = np.percentile(predictions_test_homo, 95, axis=0)
hetero_ci_5 = np.percentile(predictions_test_hetero, 5, axis=0)
hetero_ci_95 = np.percentile(predictions_test_hetero, 95, axis=0)

coverage_homo = np.mean((y_test >= homo_ci_5) & (y_test <= homo_ci_95))
coverage_hetero = np.mean((y_test >= hetero_ci_5) & (y_test <= hetero_ci_95))

plt.bar(['Homoscedastic', 'Heteroscedastic'], [coverage_homo, coverage_hetero],
        color=['blue', 'red'], alpha=0.6)
plt.axhline(y=0.9, color='k', linestyle='--', label='Expected 90% Coverage')
plt.ylabel('Coverage Rate')
plt.title('90% Prediction Interval Coverage')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Prediction uncertainty comparison
plt.subplot(2, 2, 4)
homo_uncertainty = np.std(predictions_test_homo, axis=0)
hetero_uncertainty = np.std(predictions_test_hetero, axis=0)

plt.scatter(homo_uncertainty, hetero_uncertainty, alpha=0.6)
plt.plot([homo_uncertainty.min(), homo_uncertainty.max()],
         [homo_uncertainty.min(), homo_uncertainty.max()], 'k--', alpha=0.5, label='Equal Uncertainty')
plt.xlabel('Homoscedastic Uncertainty (std)')
plt.ylabel('Heteroscedastic Uncertainty (std)')
plt.title('Prediction Uncertainty Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out_of_sample_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print("All plots saved as PNG files in the current directory.")
print("Check the generated figures for detailed model comparison results.")