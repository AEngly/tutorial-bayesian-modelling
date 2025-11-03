import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Beta parameter pairs
beta_params = [
    (1, 1),
    (2, 2),
    (5, 5),
    (10, 10),
    (20, 20),
    (50, 50),
    (500, 500)
]

# Generate x-values
x = np.linspace(0, 1, 500)

# Plot
plt.figure(figsize=(8, 5))
for a, b in beta_params:
    y = beta.pdf(x, a, b)
    plt.plot(x, y, label=f"Beta({a},{b})")

plt.title("Beta distributions: Increasing parameters together decrease variance", pad=20)
plt.xlabel("Probability of heads (p)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("beta_distributions.png", dpi=300)
plt.show()