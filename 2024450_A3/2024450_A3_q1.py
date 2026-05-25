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
y0 = (y_train==0).astype(int)
y1 = (y_train==1).astype(int)
y2 = (y_train==2).astype(int)

y0_test = (y_test==0).astype(int)
y1_test = (y_test==1).astype(int)
y2_test = (y_test==2).astype(int)

lamdas = np.array([1/10000, 1/1000, 1/100, 1/10, 1, 10, 100])

train_errors = []
test_errors = []
ridge_paths = []

for lamda in lamdas:
    w0 = np.linalg.inv(x_train.T @ x_train + lamda*np.eye(x_train.shape[1])) @ x_train.T @ y0
    w1 = np.linalg.inv(x_train.T @ x_train + lamda*np.eye(x_train.shape[1])) @ x_train.T @ y1
    w2 = np.linalg.inv(x_train.T @ x_train + lamda*np.eye(x_train.shape[1])) @ x_train.T @ y2
    ridge_paths.append(w0)

    pred_train0 = x_train @ w0
    pred_train1 = x_train @ w1
    pred_train2 = x_train @ w2
    # print(pred_train0)
    # print(pred_train1)
    # print(pred_train2)


    pred_test0 = x_test @ w0
    pred_test1 = x_test @ w1
    pred_test2 = x_test @ w2
    # print(pred_test0)
    # print(pred_test1)
    # print(pred_test2)

    train_mse = (np.mean((pred_train0 - y0) ** 2) + np.mean((pred_train1 - y1) ** 2) + np.mean((pred_train2 - y2) ** 2)) / 3
    test_mse = (np.mean((pred_test0 - y0_test) ** 2) + np.mean((pred_test1 - y1_test) ** 2) + np.mean((pred_test2 - y2_test) ** 2)) / 3

    train_errors.append(train_mse)
    test_errors.append(test_mse)

best_ridge_lambda = lamdas[np.argmin(test_errors)]

plt.plot(lamdas, train_errors, marker='o', label='Train MSE')
plt.plot(lamdas, test_errors, marker='s', label='Test MSE')

plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Ridge: Train/Test MSE vs Lambda")
plt.legend()
plt.grid(True)
plt.show()


# lasso regression 
from sklearn.linear_model import Lasso

lasso_train_mse = []
lasso_test_mse = []
non_zero_counts = []
lasso_paths = []

for lamda in lamdas:

    model0 = Lasso(alpha=lamda, fit_intercept=False, max_iter=10000)
    model1 = Lasso(alpha=lamda, fit_intercept=False, max_iter=10000)
    model2 = Lasso(alpha=lamda, fit_intercept=False, max_iter=10000)

    model0.fit(x_train, y0)
    model1.fit(x_train, y1)
    model2.fit(x_train, y2)

    w0 = model0.coef_
    w1 = model1.coef_
    w2 = model2.coef_

    lasso_paths.append(w0)

    count0 = np.sum(w0 != 0)
    count1 = np.sum(w1 != 0)
    count2 = np.sum(w2 != 0)

    avg = (count0 + count1 + count2) / 3
    non_zero_counts.append(avg)

    pred_train0 = x_train @ w0
    pred_train1 = x_train @ w1
    pred_train2 = x_train @ w2

    pred_test0 = x_test @ w0
    pred_test1 = x_test @ w1
    pred_test2 = x_test @ w2

    train_mse = (np.mean((pred_train0 - y0) ** 2) +np.mean((pred_train1 - y1) ** 2) +np.mean((pred_train2 - y2) ** 2))/3
    test_mse = (np.mean((pred_test0 - y0_test) ** 2) +np.mean((pred_test1 - y1_test) ** 2) +np.mean((pred_test2 - y2_test) ** 2))/3

    lasso_train_mse.append(train_mse)
    lasso_test_mse.append(test_mse)

best_lasso_lambda = lamdas[np.argmin(lasso_test_mse)]

plt.plot(lamdas, lasso_train_mse, marker='o', label='Train MSE')
plt.plot(lamdas, lasso_test_mse, marker='s', label='Test MSE')

plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Lasso Regression MSE vs Lambda")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(lamdas, non_zero_counts, marker='o')
plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Number of Non-Zero Coefficients")
plt.title(" lambda vs non-negative coefficients")
plt.grid(True)
plt.show()


ridge_paths = np.array(ridge_paths)
plt.figure(figsize=(8,5))

for i in range(x_train.shape[1]):
    plt.plot(lamdas, ridge_paths[:, i])

plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regularization Path (Class 0)")
plt.grid(True)
plt.show()


lasso_paths = np.array(lasso_paths)
plt.figure(figsize=(8,5))

for i in range(x_train.shape[1]):
    plt.plot(lamdas, lasso_paths[:, i])

plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")
plt.title("Lasso Regularization Path (Class 0)")
plt.grid(True)
plt.show()


p_values = [2, 5, 10, 20, 30]
train_errors = []
test_errors = []

for p in p_values:
    s = (x_train_original.T @ x_train_original)/(x_train_original.shape[0]-1)
    x_train_p, x_test_p = PCA(s, p, x_train_original, x_test_original)

    w0 = np.linalg.inv(x_train_p.T @ x_train_p + best_ridge_lambda*np.eye(x_train_p.shape[1])) @ x_train_p.T @ y0
    w1 = np.linalg.inv(x_train_p.T @ x_train_p + best_ridge_lambda*np.eye(x_train_p.shape[1])) @ x_train_p.T @ y1
    w2 = np.linalg.inv(x_train_p.T @ x_train_p + best_ridge_lambda*np.eye(x_train_p.shape[1])) @ x_train_p.T @ y2

    pred_train0 = x_train_p @ w0
    pred_train1 = x_train_p @ w1
    pred_train2 = x_train_p @ w2

    pred_test0 = x_test_p @ w0
    pred_test1 = x_test_p @ w1
    pred_test2 = x_test_p @ w2

    train_errors_p = (np.mean((pred_train0 - y0) ** 2) +np.mean((pred_train1 - y1) ** 2) +np.mean((pred_train2 - y2) ** 2)) / 3

    test_errors_p = (np.mean((pred_test0 - y0_test) ** 2) +np.mean((pred_test1 - y1_test) ** 2) +np.mean((pred_test2 - y2_test) ** 2)) / 3

    train_errors.append(train_errors_p)
    test_errors.append(test_errors_p)

plt.plot(p_values, train_errors, marker='o', label='Train MSE')
plt.plot(p_values, test_errors, marker='s', label='Test MSE')

plt.xlabel("Number of PCA components (p)")
plt.ylabel("MSE")
plt.title("Error vs Ridge model Complexity")
plt.legend()
plt.grid(True)
plt.show()

train_errors = []
test_errors = []
for p in p_values:
    s = (x_train_original.T @ x_train_original)/(x_train_original.shape[0]-1)
    x_train_p, x_test_p = PCA(s, p, x_train_original, x_test_original)


    model0 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)
    model1 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)
    model2 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)

    model0.fit(x_train_p, y0)
    model1.fit(x_train_p, y1)
    model2.fit(x_train_p, y2)

    w0 = model0.coef_
    w1 = model1.coef_
    w2 = model2.coef_

    pred_train0 = x_train_p @ w0
    pred_train1 = x_train_p @ w1
    pred_train2 = x_train_p @ w2

    pred_test0 = x_test_p @ w0
    pred_test1 = x_test_p @ w1
    pred_test2 = x_test_p @ w2


    train_errors_p = (np.mean((pred_train0 - y0) ** 2) +np.mean((pred_train1 - y1) ** 2) +np.mean((pred_train2 - y2) ** 2)) / 3

    test_errors_p = (np.mean((pred_test0 - y0_test) ** 2) +np.mean((pred_test1 - y1_test) ** 2) +np.mean((pred_test2 - y2_test) ** 2)) / 3

    train_errors.append(train_errors_p)
    test_errors.append(test_errors_p)

plt.plot(p_values, train_errors, marker='o', label='Train MSE')
plt.plot(p_values, test_errors, marker='s', label='Test MSE')

plt.xlabel("Number of PCA components (p)")
plt.ylabel("MSE")
plt.title("Error vs lasso model Complexity")
plt.legend()
plt.grid(True)
plt.show()


#Accuracy comparison

w0 = np.linalg.inv(x_train.T @ x_train + best_ridge_lambda*np.eye(x_train.shape[1])) @ x_train.T @ y0
w1 = np.linalg.inv(x_train.T @ x_train + best_ridge_lambda*np.eye(x_train.shape[1])) @ x_train.T @ y1
w2 = np.linalg.inv(x_train.T @ x_train + best_ridge_lambda*np.eye(x_train.shape[1])) @ x_train.T @ y2

pred0 = x_test @ w0
pred1 = x_test @ w1
pred2 = x_test @ w2

scores = np.array([pred0, pred1, pred2])
ridge_pred = np.argmax(scores, axis=0)

ridge_accuracy = np.mean(ridge_pred == y_test) * 100
print("Best Ridge Lambda:", best_ridge_lambda)
print("Ridge Test Accuracy:", ridge_accuracy)


model0 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)
model1 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)
model2 = Lasso(alpha=best_lasso_lambda, fit_intercept=False, max_iter=10000)

model0.fit(x_train, y0)
model1.fit(x_train, y1)
model2.fit(x_train, y2)

pred0 = x_test @ model0.coef_
pred1 = x_test @ model1.coef_
pred2 = x_test @ model2.coef_

scores = np.array([pred0, pred1, pred2])
lasso_pred = np.argmax(scores, axis=0)

lasso_accuracy = np.mean(lasso_pred == y_test) * 100
print("Best Lasso Lambda:", best_lasso_lambda)
print("Lasso Test Accuracy:", lasso_accuracy)
