import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_data(cov_scale=1):

    mean_neg = [-3, -3]
    mean_pos = [3, 3]

    cov = cov_scale * np.eye(2)

    X_neg = np.random.multivariate_normal(mean_neg, cov, 200)
    X_pos = np.random.multivariate_normal(mean_pos, cov, 200)

    X = np.vstack((X_neg, X_pos))
    y = np.hstack((-1*np.ones(200), np.ones(200)))

    return X, y


def split_data(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(0.7 * len(X))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def perceptron_train(X, y, eta=0.01, max_epochs=300):

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0

    errors_per_epoch = []

    for epoch in range(max_epochs):

        errors = 0

        for i in range(n_samples):

            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w = w + eta * y[i] * X[i]
                b = b + eta * y[i]
                errors += 1

        errors_per_epoch.append(errors)

        # Early stopping
        if errors == 0:
            print(f"Converged at epoch {epoch+1}")
            break

    return w, b, errors_per_epoch


def predict(X, w, b):
    return np.sign(X @ w + b)


def plot_boundary(X, y, w, b, title):

    plt.figure()

    # scatter
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], label="Class -1")
    plt.scatter(X[y==1][:,0], X[y==1][:,1], label="Class +1")

    # decision boundary
    x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    y_vals = -(w[0]*x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k--')

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()



X_A, y_A = generate_data(cov_scale=1)
X_train_A, y_train_A, X_test_A, y_test_A = split_data(X_A, y_A)

w_A, b_A, errors_A = perceptron_train(X_train_A, y_train_A)

# Plot errors
plt.plot(errors_A)
plt.title("Dataset A: Errors per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Misclassified")
plt.show()

# Plot boundary
plot_boundary(X_train_A, y_train_A, w_A, b_A, "Dataset A Decision Boundary (Train)")

# Test accuracy
preds_A = predict(X_test_A, w_A, b_A)
acc_A = np.mean(preds_A == y_test_A)
print("Dataset A Test Accuracy:", acc_A)


# ----------------------------
# RUN FOR DATASET B
# ----------------------------
X_B, y_B = generate_data(cov_scale=3)
X_train_B, y_train_B, X_test_B, y_test_B = split_data(X_B, y_B)

w_B, b_B, errors_B = perceptron_train(X_train_B, y_train_B)

# Plot errors
plt.plot(errors_B)
plt.title("Dataset B: Errors per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Misclassified")
plt.show()

# Plot boundary
plot_boundary(X_train_B, y_train_B, w_B, b_B, "Dataset B Decision Boundary (Train)")

# Test accuracy
preds_B = predict(X_test_B, w_B, b_B)
acc_B = np.mean(preds_B == y_test_B)
print("Dataset B Test Accuracy:", acc_B)
