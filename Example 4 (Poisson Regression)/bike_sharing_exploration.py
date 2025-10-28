"""

Title: Bike Sharing Demand Forecasting
Version: 1.0

Date: 14th October 2025
Authors: Andreas Engly (ANENG) and Anders Runge Walther (AWR)

"""

#######################################################################
####################         1. IMPORTS          ######################
#######################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, nbinom

#######################################################################
####################       2. LOAD DATA          ######################
#######################################################################

df_day = pd.read_csv("./bike+sharing+dataset/day.csv")
df_hour = pd.read_csv("./bike+sharing+dataset/hour.csv")

# Print message to user
print("\n"*5, """
#######################################################################
####################       2. LOAD DATA          ######################
#######################################################################
""")

# Print summary of dataframes
print("\n", "SUMMARY OF DF_DAY:", "\n")
print(df_day.describe())
print(df_day.info())
print(df_day.head())
print("\n", "SUMMARY OF DF_HOUR:", "\n")
print(df_hour.describe())
print(df_hour.info())
print(df_hour.head())

#######################################################################
##########  3. VISUALIZATION OF TARGET AND FEATURES          ##########
#######################################################################

# Print message to user
print("\n"*5, """
#######################################################################
##########  3. VISUALIZATION OF TARGET AND FEATURES          ##########
#######################################################################
""")   

# -------- Figure 1: Histogram of target variable 'cnt' with Poisson fit -----------

# Extract the count data
count_data = df_hour['cnt']
mean_cnt = count_data.mean()
var_cnt = count_data.var()

# Overdispersion statistic
overdispersion_stat = var_cnt / mean_cnt

# Print stats
print(f"Mean of 'cnt': {mean_cnt:.2f}")
print(f"Variance of 'cnt': {var_cnt:.2f}")
print(f"Overdispersion (variance / mean): {overdispersion_stat:.2f}")

# Range of x-values for PMFs
x = np.arange(0, count_data.max() + 1)

# --- Poisson PMF ---
poisson_pmf = poisson.pmf(x, mu=mean_cnt)

# --- Negative Binomial Estimation (Method of Moments) ---
# NB: variance = mu + (mu^2 / r) ⇒ solve for r
r = (mean_cnt ** 2) / (var_cnt - mean_cnt)
p = r / (r + mean_cnt)
nbinom_pmf = nbinom.pmf(x, r, p)

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(12, 8))
plt.hist(count_data, bins=30, density=True, alpha=0.6, color='g', label='Observed Data')
plt.plot(x, nbinom_pmf, 'ro-', label=f'Negative Binomial Distribution (r = {r:.1f}, p = {p:.2f})', markersize=4)
plt.title(f'Histogram of Hourly Bike Rentals', pad=15)
plt.xlabel('Total Bike Rentals [#]')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("bike_rentals_histogram.png", dpi=300)
plt.show()

# --------- Figure 2: Scatter plots of continuous features vs target variable 'cnt' -----------

# Explore the relationship between features and cnt
continuous_cols = ['temp', 'atemp', 'hum', 'windspeed']

# Set up subplots grid
n_cols = 2
n_rows = int(len(continuous_cols) / n_cols + 0.5)

# Create the figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()  # Flatten in case of single row

# Plot each scatter plot in a subplot
for i, col in enumerate(continuous_cols):
    sns.scatterplot(x=col, y='cnt', data=df_hour, alpha=0.3, ax=axes[i])
    axes[i].set_title(f'cnt vs {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('cnt')

# Remove unused subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("bike_rental_continuous_features.png", dpi=300)
plt.show()

# --------- Figure 3: Boxplots of categorical features vs target variable 'cnt' -----------

# Define your categorical features (should be coded as categories or integers)
categorical_cols = ['season', 'mnth', 'weekday', 'hr', 'weathersit', 'holiday', 'workingday']

# Set up subplot grid
n_cols = 2
n_rows = int(len(categorical_cols) / n_cols + 0.5)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))
axes = axes.flatten()

# Plot each boxplot
for i, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y='cnt', data=df_hour, ax=axes[i])
    axes[i].set_title(f'cnt by {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('cnt')

# Remove unused subplots if necessary
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("bike_rental_discrete_features.png", dpi=300)
plt.show()