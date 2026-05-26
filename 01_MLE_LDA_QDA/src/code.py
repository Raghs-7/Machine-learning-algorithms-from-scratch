import numpy as np
import struct
from os.path import join
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import kagglehub      


# MNIST Data Loader
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        with open(labels_filepath, 'rb') as file:
            struct.unpack(">II", file.read(8))
            labels = np.frombuffer(file.read(), dtype=np.uint8)
        
        with open(images_filepath, 'rb') as file:
            struct.unpack(">IIII", file.read(16))
            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(-1, 28, 28)
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", path)

input_path = path
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                   test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Keep only digits 0,1,2
train_mask = np.isin(y_train, [0,1,2])
test_mask  = np.isin(y_test,  [0,1,2])

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test,  y_test  = x_test[test_mask],  y_test[test_mask]

# Sample 100 per class
np.random.seed(42)

def sample_100(x, y, cls):
    xc, yc = x[y == cls], y[y == cls]
    idx = np.random.choice(len(xc), 100, replace=False)
    return xc[idx], yc[idx]

x_train_0, y_train_0 = sample_100(x_train, y_train, 0)
x_train_1, y_train_1 = sample_100(x_train, y_train, 1)
x_train_2, y_train_2 = sample_100(x_train, y_train, 2)

x_test_0, y_test_0 = sample_100(x_test, y_test, 0)
x_test_1, y_test_1 = sample_100(x_test, y_test, 1)
x_test_2, y_test_2 = sample_100(x_test, y_test, 2)

x_test_all = np.concatenate([x_test_0, x_test_1, x_test_2])
y_test_all = np.array([0]*100 + [1]*100 + [2]*100)

# Preprocess
def preprocess(x):
    x = x.astype(np.float32) / 255.0
    return x.reshape(x.shape[0], -1).T

X0, X1, X2 = preprocess(x_train_0), preprocess(x_train_1), preprocess(x_train_2)
X_test = preprocess(x_test_all)

# Means and Covariances

mu0 = np.mean(X0, axis=1, keepdims=True)
mu1 = np.mean(X1, axis=1, keepdims=True)
mu2 = np.mean(X2, axis=1, keepdims=True)

def cov(X, mu):
    Xc = X - mu
    return (Xc @ Xc.T) / (X.shape[1] - 1)

Sigma0, Sigma1, Sigma2 = cov(X0, mu0), cov(X1, mu1), cov(X2, mu2)

eps = 1e-2
Sigma0 += eps*np.eye(784)
Sigma1 += eps*np.eye(784)
Sigma2 += eps*np.eye(784)

P0 = P1 = P2 = 1/3

# QDA (Fixed with slogdet)
S0_inv, S1_inv, S2_inv = np.linalg.inv(Sigma0), np.linalg.inv(Sigma1), np.linalg.inv(Sigma2)
_, logdet0 = np.linalg.slogdet(Sigma0)
_, logdet1 = np.linalg.slogdet(Sigma1)
_, logdet2 = np.linalg.slogdet(Sigma2)

def g_QDA_fixed(x, mu, S_inv, logdetS):
    d = x - mu
    return (-0.5 * d.T @ S_inv @ d - 0.5*logdetS)

def predict_QDA_fixed(x):
    x = x.reshape(-1,1)
    return np.argmax([
        g_QDA_fixed(x, mu0, S0_inv, logdet0),
        g_QDA_fixed(x, mu1, S1_inv, logdet1),
        g_QDA_fixed(x, mu2, S2_inv, logdet2)
    ])

pred_QDA_fixed = np.array([predict_QDA_fixed(X_test[:,i]) for i in range(X_test.shape[1])])
print("QDA (Fixed) Accuracy:", np.mean(pred_QDA_fixed == y_test_all)*100, "%")

# LDA
Sigma_shared = (Sigma0 + Sigma1 + Sigma2)/3 + eps*np.eye(784)
S_inv = np.linalg.inv(Sigma_shared)

def g_LDA(x, mu, S_inv):
    return x.T @ S_inv @ mu - 0.5*mu.T @ S_inv @ mu

def predict_LDA(x):
    x = x.reshape(-1,1)
    return np.argmax([
        g_LDA(x, mu0, S_inv),
        g_LDA(x, mu1, S_inv),
        g_LDA(x, mu2, S_inv)
    ])

pred_LDA = np.array([predict_LDA(X_test[:,i]) for i in range(X_test.shape[1])])
print("LDA Accuracy:", np.mean(pred_LDA == y_test_all)*100, "%")
print("\n"*5)

# t-SNE + Discriminant Visualization
X_train_all = np.hstack((X0, X1, X2)).T
y_train_all = np.array([0]*100 + [1]*100 + [2]*100)

X_train_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_train_all)
X_test_tsne  = TSNE(n_components=2, random_state=42).fit_transform(X_test.T)

sample_idx = 0
x_sample = X_test[:, sample_idx].reshape(-1,1)
true_label = y_test_all[sample_idx]

scores_LDA = [g_LDA(x_sample, mu0, S_inv).item(),
              g_LDA(x_sample, mu1, S_inv).item(),
              g_LDA(x_sample, mu2, S_inv).item()]

scores_QDA_fixed = [g_QDA_fixed(x_sample, mu0, S0_inv, logdet0).item(),
                    g_QDA_fixed(x_sample, mu1, S1_inv, logdet1).item(),
                    g_QDA_fixed(x_sample, mu2, S2_inv, logdet2).item()]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# --- Train t-SNE ---
for cls in [0, 1, 2]:
    axes[0].scatter(X_train_tsne[y_train_all == cls, 0],
                    X_train_tsne[y_train_all == cls, 1],
                    label=f"Class {cls}")
axes[0].set_title("Train t-SNE")
axes[0].legend()
axes[0].grid()

# --- Test t-SNE ---
# --- Axes have no meaning here ---
for cls in [0, 1, 2]:
    axes[1].scatter(X_test_tsne[y_test_all == cls, 0],
                    X_test_tsne[y_test_all == cls, 1],
                    label=f"Class {cls}")
axes[1].set_title("Test t-SNE")
axes[1].legend()
axes[1].grid()

# --- LDA Scores ---
axes[2].bar(['Class 0', 'Class 1', 'Class 2'], scores_LDA)
axes[2].set_title("LDA Scores")
axes[2].grid(axis='y')

# --- QDA Scores ---
axes[3].bar(['Class 0', 'Class 1', 'Class 2'], scores_QDA_fixed)
axes[3].set_title("QDA Fixed Scores")
axes[3].grid(axis='y')

plt.tight_layout()
plt.savefig("visualization.png", dpi=300, bbox_inches='tight')
print("Saved figure: visualization.png")
