import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

from dataloader import load_train_val_set

from utils import seed_everything


# def multivariate_gaussian_pdf(x, mean, cov):
#     k = len(mean)
#     det_cov = np.linalg.det(cov)
#     inv_cov = np.linalg.inv(cov)

#     # 常量前因子
#     factor = 1.0 / (np.sqrt((2 * np.pi) ** k * det_cov))

#     # 指数的上标项
#     diff = x - mean
#     exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))

#     return factor * np.exp(exponent)


# def z_score_normalization(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     X_norm = (X - mean) / std
#     return X_norm


SEED = 42
seed_everything(SEED)

X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED
)

# X_train=z_score_normalization(X_train)
# X_val=z_score_normalization(X_val)
pca = PCA(n_components=9)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_val = pca.transform(X_val)

C0_data = X_train[y_train == 0]
C1_data = X_train[y_train == 1]
C2_data = X_train[y_train == 2]
C3_data = X_train[y_train == 3]
# # for i in range(3):
# print(C0_data.shape)
# print(C1_data.shape)
# print(C2_data.shape)
# print(C3_data.shape)

C0_prior = np.sum(y_train == 0)/X_train.shape[0]
C1_prior = np.sum(y_train == 1)/X_train.shape[0]
C2_prior = np.sum(y_train == 2)/X_train.shape[0]
C3_prior = np.sum(y_train == 3)/X_train.shape[0]

# # for i in range(3):
# print(C0_prior)
# print(C1_prior)
# print(C2_prior)
# print(C3_prior)


C0_cov = np.cov(C0_data, rowvar=False)
C0_mean = np.mean(C0_data, axis=0)
C1_cov = np.cov(C1_data, rowvar=False)
C1_mean = np.mean(C1_data, axis=0)
C2_cov = np.cov(C2_data, rowvar=False)
C2_mean = np.mean(C2_data, axis=0)
C3_cov = np.cov(C3_data, rowvar=False)
C3_mean = np.mean(C3_data, axis=0)
# print(cov_matrix.shape)
# print(mean.shape)


C0_dist = multivariate_normal(C0_mean, C0_cov)  # type: ignore
C1_dist = multivariate_normal(C1_mean, C1_cov)  # type: ignore
C2_dist = multivariate_normal(C2_mean, C2_cov)  # type: ignore
C3_dist = multivariate_normal(C3_mean, C3_cov)  # type: ignore

# print(multivariate_gaussian_pdf(X_train[0],C0_mean,C0_cov))
# pdf_value = C0_dist.pdf(X_val[0])
# print(f"概率密度是: {pdf_value}")
# print(f"P(C_i|x)={pdf_value*C0_prior*X_train.shape[0]}")

# input_set = X_val


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
