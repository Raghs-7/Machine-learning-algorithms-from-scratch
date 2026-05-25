import numpy as np
import matplotlib.pyplot as plt

data = np.load('mnist.npz')

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

mask = np.isin(y_train, [0, 1, 2])
x_train = x_train[mask]
y_train = y_train[mask]

mask = np.isin(y_test, [0, 1, 2])
x_test = x_test[mask]
y_test = y_test[mask]


x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

x_train = x_train/255.0
x_train_original = x_train.copy()
x_test = x_test/255.0
x_test_original = x_test.copy()

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



# print(y_train)
x_train, x_test = PCA(s, 10, x_train, x_test);


def gini(y):
    impurity = 1.0

    for c in range(3):
        pk = np.sum(y == c) / len(y)
        impurity -= pk ** 2

    return impurity


def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float('inf')

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

            wg = (len(y_l)/len(y))*gini(y_l) + (len(y_r)/len(y))*gini(y_r)

            if wg < best_gini:
                best_gini = wg
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


root_feature, root_threshold, root_gini = best_split(x_train, y_train)

print("Best Feature:", root_feature)
print("Best Threshold:", root_threshold)
print("Best Gini:", root_gini)
print()

left_mask = x_train[:, root_feature] <= root_threshold
right_mask = x_train[:, root_feature] > root_threshold

X_left = x_train[left_mask]
y_left = y_train[left_mask]

X_right = x_train[right_mask]
y_right = y_train[right_mask]

best_feature_left, best_threshold_left, best_gini_left = best_split(X_left, y_left)
print("Best split in LEFT child")
print("Feature:", best_feature_left)
print("Threshold:", best_threshold_left)
print("WG:", best_gini_left)

best_feature_right, best_threshold_right, best_gini_right = best_split(X_right, y_right)
print("Best split in RIGHT child")
print("Feature:", best_feature_right)
print("Threshold:", best_threshold_right)
print("WG:", best_gini_right)


if best_gini_left < best_gini_right:
    second_node = "LEFT"
    second_feature = best_feature_left
    second_threshold = best_threshold_left
else:
    second_node = "RIGHT"
    second_feature = best_feature_right
    second_threshold = best_threshold_right

print()
print("Second split chosen on:", second_node)
print("Feature:", second_feature)
print("Threshold:", second_threshold)
print()


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]



if second_node == "LEFT":
    left_left_mask = X_left[:, second_feature] <= second_threshold
    left_right_mask = X_left[:, second_feature] > second_threshold

    y_leaf1 = y_left[left_left_mask]
    y_leaf2 = y_left[left_right_mask]
    y_leaf3 = y_right

else:
    right_left_mask = X_right[:, second_feature] <= second_threshold
    right_right_mask = X_right[:, second_feature] > second_threshold

    y_leaf1 = y_left
    y_leaf2 = y_right[right_left_mask]
    y_leaf3 = y_right[right_right_mask]

leaf1_label = majority_class(y_leaf1)
leaf2_label = majority_class(y_leaf2)
leaf3_label = majority_class(y_leaf3)

def predict_sample(x):
    if x[root_feature] <= root_threshold:

        if second_node == "LEFT":
            if x[second_feature] <= second_threshold:
                return leaf1_label
            else:
                return leaf2_label
        else:
            return leaf1_label

    else:
        if second_node == "RIGHT":
            if x[second_feature] <= second_threshold:
                return leaf2_label
            else:
                return leaf3_label
        else:
            return leaf3_label

y_pred = np.array([predict_sample(x) for x in x_test])
overall_accuracy = np.mean(y_pred == y_test)
print("Overall Test Accuracy:", overall_accuracy)
print()


for c in [0, 1, 2]:
    class_mask = y_test == c
    class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
    print(f"Class {c} Accuracy:", class_acc)

print()


def build_tree(X, y):
    root_feature, root_threshold, root_gini = best_split(X, y)

    left_mask = X[:, root_feature] <= root_threshold
    right_mask = X[:, root_feature] > root_threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    f_left, t_left, g_left = best_split(X_left, y_left)
    f_right, t_right, g_right = best_split(X_right, y_right)

    if g_left < g_right:
        second_node = "LEFT"
        second_feature = f_left
        second_threshold = t_left

        ll_mask = X_left[:, second_feature] <= second_threshold
        lr_mask = X_left[:, second_feature] > second_threshold

        leaf1 = majority_class(y_left[ll_mask])
        leaf2 = majority_class(y_left[lr_mask])
        leaf3 = majority_class(y_right)

    else:
        second_node = "RIGHT"
        second_feature = f_right
        second_threshold = t_right

        rl_mask = X_right[:, second_feature] <= second_threshold
        rr_mask = X_right[:, second_feature] > second_threshold

        leaf1 = majority_class(y_left)
        leaf2 = majority_class(y_right[rl_mask])
        leaf3 = majority_class(y_right[rr_mask])

    return {
        "root_feature": root_feature,
        "root_threshold": root_threshold,
        "second_node": second_node,
        "second_feature": second_feature,
        "second_threshold": second_threshold,
        "leaf1": leaf1,
        "leaf2": leaf2,
        "leaf3": leaf3
    }


def predict_sample(tree, x):
    rf = tree["root_feature"]
    rt = tree["root_threshold"]
    sn = tree["second_node"]
    sf = tree["second_feature"]
    st = tree["second_threshold"]

    if x[rf] <= rt:
        if sn == "LEFT":
            if x[sf] <= st:
                return tree["leaf1"]
            else:
                return tree["leaf2"]
        else:
            return tree["leaf1"]
    else:
        if sn == "RIGHT":
            if x[sf] <= st:
                return tree["leaf2"]
            else:
                return tree["leaf3"]
        else:
            return tree["leaf3"]

def best_split_rf(X, y, k):
    best_feature = None
    best_threshold = None
    best_gini = float('inf')

    # randomly choose k features out of p
    selected_features = np.random.choice(X.shape[1], size=k, replace=False)

    for feature in selected_features:
        values = np.sort(np.unique(X[:, feature]))

        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            wg = (len(y_left)/len(y))*gini(y_left) + \
                 (len(y_right)/len(y))*gini(y_right)

            if wg < best_gini:
                best_gini = wg
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gini

def build_rf_tree(X, y, k):
    root_feature, root_threshold, root_gini = best_split_rf(X, y, k)

    left_mask = X[:, root_feature] <= root_threshold
    right_mask = X[:, root_feature] > root_threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    f_left, t_left, g_left = best_split_rf(X_left, y_left, k)
    f_right, t_right, g_right = best_split_rf(X_right, y_right, k)

    if g_left < g_right:
        second_node = "LEFT"
        second_feature = f_left
        second_threshold = t_left

        ll_mask = X_left[:, second_feature] <= second_threshold
        lr_mask = X_left[:, second_feature] > second_threshold

        leaf1 = majority_class(y_left[ll_mask])
        leaf2 = majority_class(y_left[lr_mask])
        leaf3 = majority_class(y_right)

    else:
        second_node = "RIGHT"
        second_feature = f_right
        second_threshold = t_right

        rl_mask = X_right[:, second_feature] <= second_threshold
        rr_mask = X_right[:, second_feature] > second_threshold

        leaf1 = majority_class(y_left)
        leaf2 = majority_class(y_right[rl_mask])
        leaf3 = majority_class(y_right[rr_mask])

    return {
        "root_feature": root_feature,
        "root_threshold": root_threshold,
        "second_node": second_node,
        "second_feature": second_feature,
        "second_threshold": second_threshold,
        "leaf1": leaf1,
        "leaf2": leaf2,
        "leaf3": leaf3
    }



oob_errors = []
oobs = []
trees = []
rf_trees = []
k = 3

for i in range(5):

    bootstrap_indices = np.random.choice(len(x_train), size=len(x_train),replace=True)

    x_boot = x_train[bootstrap_indices]
    y_boot = y_train[bootstrap_indices]

    rf_tree = build_rf_tree(x_boot, y_boot, k)
    rf_trees.append(rf_tree)

    all_indices = np.arange(len(x_train))
    oob_indices = np.setdiff1d(all_indices, bootstrap_indices)
    oobs.append(oob_indices)

    X_oob = x_train[oob_indices]
    y_oob = y_train[oob_indices]

    tree = build_tree(x_boot, y_boot)
    trees.append(tree)

    if len(X_oob) == 0:
        continue

    y_pred_oob = np.array([predict_sample(tree, x) for x in X_oob])

    oob_error = 1 - np.mean(y_pred_oob == y_oob)
    oob_errors.append(oob_error)

    print(f"Tree {i+1} OOB Error:", oob_error)


avg_oob_error = np.mean(oob_errors)
print("\nAverage OOB Error across 5 trees:", avg_oob_error)
print()

y_pred_test = []
for x in x_test:
    preds = [predict_sample(tree, x) for tree in trees]
    y_pred_test.append(majority_class(np.array(preds)))

y_pred_test = np.array(y_pred_test)
overall_acc = np.mean(y_pred_test == y_test)
print("Bagging Test Accuracy:", overall_acc)
print()

for c in [0, 1, 2]:
    mask = y_test == c
    class_acc = np.mean(y_pred_test[mask] == y_test[mask])
    print(f"Bagging Class {c} Accuracy:", class_acc)

print()


oob_votes = {i: [] for i in range(len(x_train))}
for tree_idx in range(5):
    rf_tree = rf_trees[tree_idx]
    oob_indices = oobs[tree_idx]

    for idx in oob_indices:
        pred = predict_sample(rf_tree, x_train[idx])
        oob_votes[idx].append(pred)

oob_pred = []
oob_true = []

for idx in range(len(x_train)):
    if len(oob_votes[idx]) == 0:
        continue

    pred = majority_class(np.array(oob_votes[idx]))
    oob_pred.append(pred)
    oob_true.append(y_train[idx])

oob_pred = np.array(oob_pred)
oob_true = np.array(oob_true)

rf_oob_error = 1 - np.mean(oob_pred == oob_true)

print("random forest OOB Error:", rf_oob_error)
print()

y_pred_rf = []
for x in x_test:
    preds = [predict_sample(tree, x) for tree in rf_trees]
    y_pred_rf.append(majority_class(np.array(preds)))

y_pred_rf = np.array(y_pred_rf)
print("Random Forest Test Accuracy:", np.mean(y_pred_rf == y_test))
print()