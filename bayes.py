import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from dataloader import load_train_val_set

from utils import seed_everything


SEED = 42
seed_everything(SEED)

X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED
)

pca = PCA(n_components=9)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_val = pca.transform(X_val)


C0_data = X_train[y_train == 0]
C1_data = X_train[y_train == 1]
C2_data = X_train[y_train == 2]
C3_data = X_train[y_train == 3]

C0_prior = np.sum(y_train == 0)
C1_prior = np.sum(y_train == 1)
C2_prior = np.sum(y_train == 2)
C3_prior = np.sum(y_train == 3)

C0_cov = np.cov(C0_data, rowvar=False)
C0_mean = np.mean(C0_data, axis=0)
C1_cov = np.cov(C1_data, rowvar=False)
C1_mean = np.mean(C1_data, axis=0)
C2_cov = np.cov(C2_data, rowvar=False)
C2_mean = np.mean(C2_data, axis=0)
C3_cov = np.cov(C3_data, rowvar=False)
C3_mean = np.mean(C3_data, axis=0)

C0_dist = multivariate_normal(C0_mean, C0_cov)  # type: ignore
C1_dist = multivariate_normal(C1_mean, C1_cov)  # type: ignore
C2_dist = multivariate_normal(C2_mean, C2_cov)  # type: ignore
C3_dist = multivariate_normal(C3_mean, C3_cov)  # type: ignore


ours = np.stack(
    (
        C0_dist.pdf(X_val) * C0_prior, C1_dist.pdf(X_val) * C1_prior,
        C2_dist.pdf(X_val) * C2_prior, C3_dist.pdf(X_val) * C3_prior
    ), axis=1
)
ours /= ours.sum(axis=1, keepdims=True)

target = np.zeros_like(ours)
target[np.arange(y_val.shape[0]), y_val] = 1.0

evaluation = np.sum(np.abs(ours-target))

print(evaluation)
