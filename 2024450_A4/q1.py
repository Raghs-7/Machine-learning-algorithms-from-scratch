import numpy as np
import matplotlib.pyplot as plt

data = np.load("mnist.npz")
X_train, y_train = data["x_train"], data["y_train"]
X_test, y_test = data["x_test"], data["y_test"]


# Flatten (28x28 → 784)
X_train = X_train.reshape(60000, -1)
X_test = X_test.reshape(10000, -1)

# Normalize (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0




# Train
mask_train = (y_train == 4) | (y_train == 9)
X_train = X_train[mask_train]
y_train = y_train[mask_train]

# Test
mask_test = (y_test == 4) | (y_test == 9)
X_test = X_test[mask_test]
y_test = y_test[mask_test]


y_train = np.where(y_train == 4, -1, 1)
y_test = np.where(y_test == 4, -1, 1)


idx_4 = np.where(y_train == -1)[0]  # class 4
idx_9 = np.where(y_train == 1)[0]   # class 9

np.random.shuffle(idx_4)
np.random.shuffle(idx_9)

val_idx_4 = idx_4[:1000]
val_idx_9 = idx_9[:1000]

train_idx_4 = idx_4[1000:]
train_idx_9 = idx_9[1000:]

val_idx = np.concatenate((val_idx_4, val_idx_9))
train_idx = np.concatenate((train_idx_4, train_idx_9))

np.random.shuffle(val_idx)
np.random.shuffle(train_idx)

X_val = X_train[val_idx]
y_val = y_train[val_idx]

X_train_final = X_train[train_idx]
y_train_final = y_train[train_idx]



def PCA(X_train_final, p):
    mean = np.mean(X_train_final, axis=0)
    X_train_centered = X_train_final - mean

    # Covariance matrix
    cov = np.cov(X_train_centered, rowvar=False)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # Take top p components
    W = eig_vecs[:, :p]

    return W

W = PCA(X_train_final, 5)
mean = np.mean(X_train_final, axis=0)
# Transform datasets
X_train_pca = (X_train_final - mean) @ W
X_val_pca = (X_val - mean) @ W
X_test_pca = (X_test - mean) @ W




n_samples, n_features = X_train_pca.shape

# uniform weights
weights = np.ones(n_samples) / n_samples

stumps = []      # store (feature, threshold, polarity)
alphas = []      # store alpha values


T = 300
n_samples, n_features = X_train_pca.shape

for t in range(T):

    best_feature = -1
    best_threshold = 0
    best_polarity = 1
    min_error = float('inf')

    # --- find best stump ---
    for j in range(n_features):

        X_j = X_train_pca[:, j]

        sorted_idx = np.argsort(X_j)
        X_sorted = X_j[sorted_idx]

        # midpoints
        midpoints = (X_sorted[:-1] + X_sorted[1:]) / 2

        # pick 1000 thresholds
        if len(midpoints) > 1000:
            idx = np.random.choice(len(midpoints), 1000, replace=False)
            thresholds = midpoints[idx]
        else:
            thresholds = midpoints

        # try splits
        for t_val in thresholds:
            for polarity in [1, -1]:

                predictions = np.ones(n_samples)

                if polarity == 1:
                    predictions[X_j < t_val] = -1
                else:
                    predictions[X_j < t_val] = 1

                error = np.sum(weights * (predictions != y_train_final))

                if error < min_error:
                    min_error = error
                    best_feature = j
                    best_threshold = t_val
                    best_polarity = polarity

    epsilon = max(min_error, 1e-10) # to avoid zero
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)

    predictions = np.ones(n_samples)
    X_j = X_train_pca[:, best_feature]

    if best_polarity == 1:
        predictions[X_j < best_threshold] = -1
    else:
        predictions[X_j < best_threshold] = 1

    weights = weights * np.exp(-alpha * y_train_final * predictions)
    weights = weights / np.sum(weights)

    stumps.append((best_feature, best_threshold, best_polarity))
    alphas.append(alpha)





val_scores = np.zeros(X_val_pca.shape[0])  # running sum
val_accuracies = []

for t in range(T):

    feature, threshold, polarity = stumps[t]
    alpha = alphas[t]

    preds = np.ones(X_val_pca.shape[0])

    if polarity == 1:
        preds[X_val_pca[:, feature] < threshold] = -1
    else:
        preds[X_val_pca[:, feature] < threshold] = 1

    # add to running sum
    val_scores += alpha * preds

    # final prediction
    final_preds = np.sign(val_scores)

    # accuracy
    acc = np.mean(final_preds == y_val)

    val_accuracies.append(acc)

plt.plot(range(1, T+1), val_accuracies)
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Number of Trees")
plt.grid()

plt.savefig("val_accuracy_plot.png")
plt.show()


best_iter = np.argmax(val_accuracies) + 1   # +1 because index starts from 0
print("Best number of trees:", best_iter)


test_scores = np.zeros(X_test_pca.shape[0])
for t in range(best_iter):
    
    feature, threshold, polarity = stumps[t]
    alpha = alphas[t]

    preds = np.ones(X_test_pca.shape[0])

    if polarity == 1:
        preds[X_test_pca[:, feature] < threshold] = -1
    else:
        preds[X_test_pca[:, feature] < threshold] = 1

    test_scores += alpha * preds

# final prediction
final_preds = np.sign(test_scores)

# accuracy
test_accuracy = np.mean(final_preds == y_test)

print("Test Accuracy:", test_accuracy)