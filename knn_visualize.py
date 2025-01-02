import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from dataloader import load_train_val_set
from dataloader import load_train_val_set, load_test_set, save_results
from metrics import get_tianchi_metric

from utils import seed_everything
import time
import pandas as pd

# Set random seed
SEED = 42
seed_everything(SEED)


X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED)


n_components_list = [2, 4, 6, 8, 9,  10, 12]
n_neighbors_list = [1, 3, 5, 7, 9]

# Storage for results
results = []


for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    for n_neighbors in n_neighbors_list:

        time_0 = time.time()

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_pca, y_train)
        time_1 = time.time()
        training_time=time_1-time_0

        ours_val = knn.predict_proba(X_val_pca)
        time_2 = time.time()
        validation_time=time_2-time_1
        y_pred = np.argmax(ours_val, axis=1)

        tc_score = get_tianchi_metric(ours_val, y_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val,y_pred, average='weighted')


        # Store the result
        results.append((n_components, n_neighbors, tc_score,training_time,validation_time,acc,f1))
        # print(f"n_components={n_components}, n_neighbors={n_neighbors}, accuracy={tc_score:.4f}")

# Convert results to numpy array for easy slicing
results = np.array(results)

# Plot results
fig = plt.figure(figsize=(20, 7))
ax = fig.add_subplot(131, projection='3d')

# Scatter plot
ax.scatter(results[:, 0], results[:, 1], results[:, 2], c='b', marker='x',label='Tianchi Score')

# Axes labels
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Number of Neighbors')
ax.set_zlabel('Tianchi score')
ax.set_title('Tianchi score vs Components and Neighbors')
ax.legend()

ax = fig.add_subplot(132, projection='3d')

ax.scatter(results[:, 0], results[:, 1], results[:, 3], c='r', marker='o',label='Training Time')
ax.scatter(results[:, 0], results[:, 1], results[:, 4], c='g', marker='^',label='Validation Time')

# Axes labels
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Number of Neighbors')
ax.set_zlabel('Time (seconds)')
ax.set_title('Time vs Components and Neighbors')
ax.legend()

ax = fig.add_subplot(133, projection='3d')

ax.scatter(results[:, 0], results[:, 1], results[:, 5], c='r', marker='o',label='Accuracy')
ax.scatter(results[:, 0], results[:, 1], results[:, 6], c='b', marker='*',label='F1_score')
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Number of Neighbors')
ax.set_zlabel('Score')
ax.set_title('Score vs Components and Neighbors')
ax.legend()
# plt.title('KNN Tianchi score for Different PCA Components and Neighbors')
plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(
    results,
    columns=['n_components', 'n_neighbors', 'tianchi_score', 'training_time', 'validation_time', 'accuracy', 'f1_score']
)

results_df.to_csv('./results/knn_visualize.csv', index=False)