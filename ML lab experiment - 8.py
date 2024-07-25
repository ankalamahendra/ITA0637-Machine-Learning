import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate sample data
np.random.seed(42)
n_samples = 500

# Parameters for the true Gaussian distributions
mu1, sigma1 = [2, 2], [[1, 0], [0, 1]]
mu2, sigma2 = [7, 7], [[1, 0], [0, 1]]

# Generate samples
data1 = np.random.multivariate_normal(mu1, sigma1, n_samples)
data2 = np.random.multivariate_normal(mu2, sigma2, n_samples)
data = np.vstack((data1, data2))

# Plot the generated data
plt.scatter(data[:, 0], data[:, 1], s=5)
plt.title("Generated Data")
plt.show()

# Number of clusters
k = 2

# Randomly initialize the parameters
np.random.seed(42)
weights = np.ones(k) / k
means = np.random.rand(k, 2) * 10
covariances = np.array([np.eye(2)] * k)

# Log likelihood
log_likelihoods = []

# EM algorithm
for iteration in range(100):
    # E-step: calculate responsibilities


    responsibilities = np.zeros((len(data), k))
    for i in range(k):
        responsibilities[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covariances[i])
    responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

    # M-step: update parameters
    N_k = responsibilities.sum(axis=0)
    weights = N_k / len(data)
    means = np.dot(responsibilities.T, data) / N_k[:, np.newaxis]
    covariances = np.zeros((k, 2, 2))
    for i in range(k):
        diff = data - means[i]
        covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / N_k[i]

    # Compute log likelihood
    log_likelihood = np.sum(np.log(np.dot(responsibilities, weights)))
    log_likelihoods.append(log_likelihood)
    if iteration > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
        break

print(f"Converged after {iteration + 1} iterations")

# Plot the log likelihood
plt.plot(log_likelihoods)
plt.title("Log Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.show()

# Plot the clustered data
colors = ['r', 'b']
for i in range(k):
    cluster = data[np.argmax(responsibilities, axis=1) == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=5, color=colors[i])

plt.title("Clustered Data")
plt.show()

print("Final parameters:")
print("Weights:", weights)
print("Means:", means)
print("Covariances:", covariances)
