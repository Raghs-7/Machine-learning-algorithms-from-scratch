import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_mask = np.isin(y_train, [0, 1, 2])
test_mask = np.isin(y_test, [0, 1, 2])
12
x_train = x_train[train_mask]
y_train = y_train[train_mask]

x_test = x_test[test_mask]
y_test = y_test[test_mask]


x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0



mean = np.mean(x_train, axis=0)
x_train = x_train - mean
x_test = x_test - mean
s = (x_train.T @ x_train)/(x_train.shape[0]-1)

def PCA(s, p, x_train, x_test):

    eigonvalues, eigenvectors = np.linalg.eigh(s)
    idx = np.argsort(eigonvalues)[::-1]
    eigonvalues = eigonvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    Up = eigenvectors[:, :p]

    x_train_p = x_train @ Up
    x_test_p = x_test @ Up

    return x_train_p, x_test_p


x_train, x_test = PCA(s, 10, x_train, x_test)

def best_split_regression(X, y):
    best_feature = None
    best_threshold = None
    best_sse = float('inf')

    for feature in range(X.shape[1]):
        values = np.sort(np.unique(X[:, feature]))
        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_l = y[left_mask]
            y_r = y[right_mask]

            left_mean = np.mean(y_l)
            right_mean = np.mean(y_r)

            left_sse = np.sum((y_l - left_mean) ** 2)
            right_sse = np.sum((y_r - right_mean) ** 2)

            total_sse = left_sse + right_sse

            if total_sse < best_sse:
                best_sse = total_sse
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_sse


def h1(X, y):
    feature, threshold, _ = best_split_regression(X, y)

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_value = np.mean(y[left_mask])

    right_value = np.mean(y[right_mask])

    stump = {
        "feature": feature,
        "threshold": threshold,
        "left_value": left_value,
        "right_value": right_value
    }

    return stump


def predict_stump(X, stump):
    feature = stump["feature"]
    threshold = stump["threshold"]
    left_value = stump["left_value"]
    right_value = stump["right_value"]

    preds = np.where( X[:, feature] <= threshold, left_value, right_value )
    return preds


model = h1(x_train, y_train)
preds = predict_stump(x_test, model)
test_mse = np.mean((y_test-preds)**2)
print("Test MSE: ", test_mse)
print()


oobs = []
stumps = []
for i in range(5):

    bootstrap_indices = np.random.choice(len(x_train), size=len(x_train),replace=True)

    x_boot = x_train[bootstrap_indices]
    y_boot = y_train[bootstrap_indices]

    all_indices = np.arange(len(x_train))
    oob_indices = np.setdiff1d(all_indices, bootstrap_indices)
    oobs.append(oob_indices)

    X_oob = x_train[oob_indices]
    y_oob = y_train[oob_indices]

    model = h1(x_boot, y_boot)
    stumps.append(model)


oob_errors = []
for i in range(5):
    oob_indices = oobs[i]
    X_oob = x_train[oob_indices]
    y_oob = y_train[oob_indices]

    all_oob_preds = []
    for model in stumps:
        preds = predict_stump(X_oob, model)
        all_oob_preds.append(preds)

    all_oob_preds = np.array(all_oob_preds)
    all_oob_preds = np.mean(all_oob_preds, axis=0)

    oob_mse = np.mean((y_oob - all_oob_preds) ** 2)
    oob_errors.append(oob_mse)

avg = np.mean(oob_errors)
print("Average OOB MSE: ", avg)
print()


single_model = h1(x_train, y_train)
single_preds = predict_stump(x_test, single_model)

bagged_preds = []

for model in stumps:
    preds = predict_stump(x_test, model)
    bagged_preds.append(preds)

bagged_preds = np.array(bagged_preds)
bagged_preds = np.mean(bagged_preds, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label="True Function", linewidth=2)
plt.plot(single_preds[:100], label="Single Stump", linestyle="--")
plt.plot(bagged_preds[:100], label="Bagged Model", linestyle=":")

plt.xlabel("Sample Index")
plt.ylabel("Prediction / True Label")
plt.title("Single Decision Stump vs Bagging")
plt.legend()
plt.grid(True)
plt.show()