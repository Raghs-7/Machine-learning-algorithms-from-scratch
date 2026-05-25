import numpy as np
import matplotlib.pyplot as plt


data = np.load("mnist.npz")

X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]


X_train = X_train.reshape(60000, -1) / 255.0
X_test = X_test.reshape(10000, -1) / 255.0


mask_train = (y_train == 4) | (y_train == 9)
mask_test = (y_test == 4) | (y_test == 9)

X_train = X_train[mask_train]
y_train = y_train[mask_train]

X_test = X_test[mask_test]
y_test = y_test[mask_test]

# Convert labels
y_train = np.where(y_train == 4, -1, 1)
y_test = np.where(y_test == 4, -1, 1)

# Split validation (1000 each)
idx_neg = np.where(y_train == -1)[0]
idx_pos = np.where(y_train == 1)[0]

np.random.shuffle(idx_neg)
np.random.shuffle(idx_pos)

val_idx = np.concatenate((idx_neg[:1000], idx_pos[:1000]))
train_idx = np.concatenate((idx_neg[1000:], idx_pos[1000:]))

X_val = X_train[val_idx]
y_val = y_train[val_idx]

X_train = X_train[train_idx]
y_train = y_train[train_idx]


mean = np.mean(X_train, axis=0)
X_centered = X_train - mean

cov = np.cov(X_centered, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov)

idx = np.argsort(eig_vals)[::-1]
W = eig_vecs[:, idx[:5]]

X_train = (X_train - mean) @ W
X_val = (X_val - mean) @ W
X_test = (X_test - mean) @ W


def find_best_stump(X, y):
    n_samples, n_features = X.shape

    best_feature = -1
    best_threshold = 0
    best_polarity = 1
    min_error = float('inf')

    for j in range(n_features):
        X_j = X[:, j]

        sorted_vals = np.sort(X_j)

        midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2

        if len(midpoints) > 1000:
            idx = np.random.choice(len(midpoints), 1000, replace=False)
            thresholds = midpoints[idx]
        else:
            thresholds = midpoints

        for t in thresholds:
            for polarity in [1, -1]:

                preds = np.ones(n_samples)

                if polarity == 1:
                    preds[X_j < t] = -1
                else:
                    preds[X_j < t] = 1

                error = np.sum(preds != y)

                if error < min_error:
                    min_error = error
                    best_feature = j
                    best_threshold = t
                    best_polarity = polarity

    return best_feature, best_threshold, best_polarity


def stump_predict(X, feature, threshold, polarity):
    preds = np.ones(len(X))
    if polarity == 1:
        preds[X[:, feature] < threshold] = -1
    else:
        preds[X[:, feature] < threshold] = 1
    return preds


etas = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
T = 300

results = {}

for eta in etas:

    F_train = np.zeros(len(X_train))
    F_val = np.zeros(len(X_val))

    stumps = []
    val_mse_list = []

    for t in range(T):

        # gradients (absolute loss)
        gradients = np.sign(y_train - F_train)

        # train stump
        f, th, pol = find_best_stump(X_train, gradients)

        stumps.append((f, th, pol))

        # update train
        preds_train = stump_predict(X_train, f, th, pol)
        F_train += eta * preds_train

        # update validation
        preds_val = stump_predict(X_val, f, th, pol)
        F_val += eta * preds_val

        # compute MSE
        mse = np.mean((y_val - F_val) ** 2)
        val_mse_list.append(mse)

    results[eta] = {
        "mse": val_mse_list,
        "stumps": stumps
    }


for eta in etas:
    plt.plot(results[eta]["mse"], label=f"eta={eta}")

plt.xlabel("Trees")
plt.ylabel("Validation MSE")
plt.title("Validation MSE vs Trees")
plt.legend()
plt.grid()
plt.show()

for eta in etas:

    mse_list = results[eta]["mse"]
    stumps = results[eta]["stumps"]

    best_iter = np.argmin(mse_list) + 1

    F_test = np.zeros(len(X_test))

    for t in range(best_iter):
        f, th, pol = stumps[t]
        preds = stump_predict(X_test, f, th, pol)
        F_test += eta * preds

    test_mse = np.mean((y_test - F_test) ** 2)

    print(f"eta={eta} | best_iter={best_iter} | test_MSE={test_mse:.4f}")