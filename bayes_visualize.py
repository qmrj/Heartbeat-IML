import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import time
from dataloader import load_train_val_set, load_test_set
from metrics import get_tianchi_metric
from utils import seed_everything
import pandas as pd

SEED = 42
seed_everything(SEED)


X_train, X_val, y_train, y_val = load_train_val_set(test_size=0.2, random_state=SEED)


n_components_list = [2, 4, 6, 8, 9, 10, 12]
accuracies = []
f1_scores = []
training_times = []
validation_times = []
tc_scores=[]

for n_components in n_components_list:

    time_0 = time.time()

   
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

   
    X_train_pca = pca.transform(X_train)


    C0_data = X_train_pca[y_train == 0]
    C1_data = X_train_pca[y_train == 1]
    C2_data = X_train_pca[y_train == 2]
    C3_data = X_train_pca[y_train == 3]

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

   
    C0_dist = multivariate_normal(C0_mean, C0_cov)
    C1_dist = multivariate_normal(C1_mean, C1_cov)
    C2_dist = multivariate_normal(C2_mean, C2_cov)
    C3_dist = multivariate_normal(C3_mean, C3_cov)

    
    time_1 = time.time()
    training_times.append(time_1 - time_0)

 
    X_val_pca = pca.transform(X_val)

    ours_val = np.stack(
        (
            C0_dist.pdf(X_val_pca) * C0_prior, 
            C1_dist.pdf(X_val_pca) * C1_prior,
            C2_dist.pdf(X_val_pca) * C2_prior, 
            C3_dist.pdf(X_val_pca) * C3_prior
        ), axis=1
    )
    ours_val /= ours_val.sum(axis=1, keepdims=True)

    time_2 = time.time()
    validation_times.append(time_2 - time_1)


    tianchi= get_tianchi_metric(ours_val, y_val)
    tc_scores.append(tianchi)


    y_pred = np.argmax(ours_val, axis=1)


    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    accuracies.append(acc)
    f1_scores.append(f1)

    # print(f"n_components={n_components} | training time: {time_1 - time_0:.2f}s | validation time: {time_2 - time_1:.2f}s")


plt.figure(figsize=(12, 6))
plt.subplot(131)  
plt.plot(n_components_list, accuracies, marker='o', label='Accuracy')
plt.plot(n_components_list, f1_scores, marker='x', label='F1 Score')
plt.title('Performance with Different PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Score')
plt.legend()
plt.grid(True)


plt.subplot(132)  
plt.plot(n_components_list, training_times, marker='o', label='Training Time')
plt.plot(n_components_list, validation_times, marker='x', label='Validation Time')
plt.title('Timing with Different PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.subplot(133)  
plt.plot(n_components_list, tc_scores, marker='o', label='Tianchi Score')
plt.title('Tianchi Score with Different PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Points')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save results to a CSV file
results = pd.DataFrame({
    'n_components': n_components_list,
    'accuracy': accuracies,
    'f1_score': f1_scores,
    'training_time': training_times,
    'validation_time': validation_times,
    'tianchi_score': tc_scores
})
results.to_csv('./results/bayes_visualize.csv', index=False)