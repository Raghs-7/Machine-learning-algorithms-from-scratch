# Machine Learning Algorithms from Scratch

## Overview

This repository contains a collection of fundamental Machine Learning algorithms implemented entirely from scratch. The project is divided into four assignments, covering topics ranging from generative models and dimensionality reduction to regularized regression, ensemble learning, and linear classifiers.

The primary objective is to understand the mathematical foundations behind these algorithms and implement them without relying on high-level machine learning libraries such as Scikit-Learn for the core logic. :contentReference[oaicite:0]{index=0}

---

## Repository Structure

```text
Machine-Learning-Algorithms-From-Scratch/
│
├── 01_MLE_LDA_QDA/
├── 02_PCA_FDA/
├── 03_Regression/
└── 04_Boosting_and_Perceptron/
```

Each assignment contains:

- Source code (`.py` and `.ipynb`)
- Experimental results
- Visualizations
- Detailed report (`report.pdf`) explaining methodology, mathematical derivations, and observations

---

# Assignment 1: Maximum Likelihood Estimation, LDA, and QDA

## Objective

Study generative classification methods and density estimation techniques.

### Dataset

- MNIST Dataset
- Classes Used: **0, 1, and 2**

### Implementations

#### Maximum Likelihood Estimation (MLE)

- Image data represented as 784-dimensional feature vectors.
- Class-specific Gaussian distributions estimated using MLE.

#### Linear Discriminant Analysis (LDA)

- Shared covariance matrix across all classes.
- Produces linear decision boundaries.
- Effective when class covariance structures are similar.

#### Quadratic Discriminant Analysis (QDA)

- Separate covariance matrix for each class.
- Produces quadratic decision boundaries.
- More flexible but more sensitive to noise.

### Visualization

- t-SNE used for reducing 784-dimensional images into 2D.
- Visual analysis of training and testing distributions.

---

# Assignment 2: PCA and Fisher Discriminant Analysis

## Objective

Explore supervised and unsupervised dimensionality reduction techniques.

### Dataset

- MNIST Dataset
- Classes Used: **0, 1, and 2**

### Implementations

#### Principal Component Analysis (PCA)

- Variance retention experiments:
  - 75%
  - 90%
  - Fixed component counts
- Reduces dimensionality while preserving maximum variance.
- Significantly lowers computational cost.

#### Fisher Discriminant Analysis (FDA)

- Maximizes:
  
  Between-Class Variance

- Minimizes:
  
  Within-Class Variance

- Produces more discriminative feature spaces than PCA.

### Key Observation

Combining:

```text
PCA → LDA/QDA
```

creates an effective pipeline for high-dimensional image classification.

---

# Assignment 3: Regression and Tree-Based Models

## Objective

Study regularized linear models and ensemble learning through Bagging.

### Datasets

#### MNIST

Classes:

- 0
- 1
- 2

#### Fashion-MNIST

Classes:

- T-shirt/Top
- Trouser
- Pullover

---

## Ridge vs Lasso Regression

### Ridge Regression (L2)

- Trained on 10-dimensional PCA projections.
- Coefficients shrink but never become exactly zero.
- Relatively insensitive to regularization parameter λ.

### Lasso Regression (L1)

- Performs automatic feature selection.
- Forces many coefficients to zero as λ increases.

### Results

| Model | Test Accuracy |
|---------|-------------|
| Ridge | 95.14% |
| Lasso | 95.14% |

---

## Decision Trees and Random Forests

### Configuration

- Maximum depth = 2
- 5 trees for ensemble methods

### Observations

- Bagging slightly outperformed Random Forest.
- Feature subsampling occasionally removed the most informative PCA component.
- In this low-dimensional setting (p = 10), Random Forest provided no significant advantage.

---

## Regression Decision Stumps

### Dataset

Fashion-MNIST

### Method

- Class labels treated as continuous targets.
- Regression stumps trained to minimize MSE.

### Observation

Bagging significantly reduced variance and improved Out-of-Bag (OOB) error compared to single regression stumps.

---

# Assignment 4: Boosting and the Perceptron

## Objective

Implement boosting algorithms and study the Rosenblatt Perceptron.

### Datasets

#### MNIST

Classes:

- 4
- 9

#### Synthetic Gaussian Data

Generated using NumPy.

---

## AdaBoost

### Configuration

- Decision Stumps
- 5-dimensional PCA representation

### Observation

- Highest validation accuracy achieved using the first stump.
- PCA compression removed important class-discriminative information.
- Additional boosting iterations provided limited benefit.

---

## Gradient Boosting

### Loss Function

Sum of Squared Residuals (SSR)

### Hyperparameter Study

Different learning rates were evaluated.

#### Small Learning Rates

- Stable convergence
- Requires more trees

#### Large Learning Rates

- Faster updates
- Risk of overshooting minima

#### Best Range

```text
0.1 – 0.2
```

Provides a good balance between convergence speed and stability.

---

## Rosenblatt Perceptron

### Experiment 1: Linearly Separable Data

- Converged in 2 epochs.
- Demonstrates the Perceptron Convergence Theorem.

### Experiment 2: Overlapping Classes

- Failed to converge.
- Oscillated indefinitely.

### Result

Despite non-convergence:

```text
Test Accuracy = 96.7%
```

Illustrating the robustness of the perceptron even when assumptions are violated.

---

# Datasets

## MNIST (Raw Binary Format)

Used in:

- Assignment 1
- Assignment 2

Dataset:

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

---

## MNIST (TensorFlow/Keras NPZ Format)

Used in:

- Assignment 3
- Assignment 4

Dataset:

https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

---

## Fashion-MNIST

Used in:

- Assignment 3

Dataset:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

---

## Synthetic Gaussian Dataset

Generated programmatically using NumPy for Assignment 4.

---

# Requirements

## Python Version

```text
Python 3.x
```

## Libraries

```bash
numpy
pandas
matplotlib
jupyter
```

Install dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

---

## Download Required Datasets

Download datasets from the links above and place them in the corresponding assignment directories.

Example:

```text
03_Regression/
│
├── mnist.npz
├── src/
└── report.pdf
```

```text
04_Boosting_and_Perceptron/
│
├── mnist.npz
├── src/
└── report.pdf
```

---

## Running Python Scripts

```bash
cd Assignment_Directory
python src/code.py
```

---

## Running Jupyter Notebooks

```bash
jupyter notebook
```

Then open the corresponding notebook:

```text
src/code.ipynb
```

---

# Reports

Each assignment includes a detailed report covering:

- Mathematical derivations
- Algorithm implementation details
- Hyperparameter tuning
- Experimental observations
- Visualizations
- Performance analysis

Refer to the `report.pdf` files inside each assignment folder for a complete discussion.

---

# Learning Outcomes

Through this repository, the following concepts are implemented and analyzed from first principles:

- Maximum Likelihood Estimation (MLE)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Principal Component Analysis (PCA)
- Fisher Discriminant Analysis (FDA)
- Ridge Regression
- Lasso Regression
- Decision Trees
- Random Forests
- Bagging
- AdaBoost
- Gradient Boosting
- Rosenblatt Perceptron

---

## Author

**Raghav Dhiman**  
B.Tech CSAI, IIIT-Delhi
