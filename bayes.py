from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from utils import seed_everything

from numpy.typing import NDArray


from dataloader import load_train_val_set

from matplotlib import pyplot as plt

from utils import seed_everything

from scipy.stats import multivariate_normal

from sklearn.decomposition import PCA


def multivariate_gaussian_pdf(x, mean, cov):
    k = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    # 常量前因子
    factor = 1.0 / (np.sqrt((2 * np.pi) ** k * det_cov))

    # 指数的上标项
    diff = x - mean
    exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))

    return factor * np.exp(exponent)


def z_score_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm


SEED = 42
seed_everything(SEED)

X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED
)

# X_train=z_score_normalization(X_train)
# X_val=z_score_normalization(X_val)
pca = PCA(n_components=9)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

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


C0_dist = multivariate_normal(C0_mean, C0_cov)
C1_dist = multivariate_normal(C1_mean, C1_cov)
C2_dist = multivariate_normal(C2_mean, C2_cov)
C3_dist = multivariate_normal(C3_mean, C3_cov)

# print(multivariate_gaussian_pdf(X_train[0],C0_mean,C0_cov))
# pdf_value = C0_dist.pdf(X_val[0])
# print(f"概率密度是: {pdf_value}")
# print(f"P(C_i|x)={pdf_value*C0_prior*X_train.shape[0]}")

input_set = X_val

evaluation=0
for i in range(X_val.shape[0]):
# for i in range(10):


    # bayes_class = 0

    # max_val = C0_dist.pdf(input_set[i])*C0_prior

    # temp = C1_dist.pdf(input_set[i])*C1_prior

    # if temp > max_val:
    #     temp = max_val
    #     bayes_class = 1

    # temp = C2_dist.pdf(input_set[i])*C2_prior

    # if temp > max_val:
    #     temp = max_val
    #     bayes_class = 2

    # temp = C3_dist.pdf(input_set[i])*C3_prior

    # if temp > max_val:
    #     temp = max_val
    #     bayes_class = 3

    factor = C0_dist.pdf(input_set[i])*C0_prior+C1_dist.pdf(input_set[i])*C1_prior + \
        C2_dist.pdf(input_set[i])*C2_prior+C3_dist.pdf(input_set[i])*C3_prior
    
    our_list= np.array([C0_dist.pdf(input_set[i])*C0_prior/factor, C1_dist.pdf(
        input_set[i])*C1_prior/factor, C2_dist.pdf(input_set[i])*C2_prior/factor,
        C3_dist.pdf(input_set[i])*C3_prior/factor])
    
    val_list=np.zeros((1,4))
    val_list[0][y_val[i]]=1
    
    # print(np.abs(our_list-val_list))

    evaluation+= np.sum(np.abs(our_list-val_list))
    
    # print(f"Bayes' classifier choose C{bayes_class} \nProbability tuple{Probability_tuple}")
print(evaluation)
