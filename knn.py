import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from dataloader import load_train_val_set, load_test_set, save_results
from metrics import print_metrics

from utils import seed_everything


SEED = 42
seed_everything(SEED)

X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED
)

# pca = PCA(n_components=9)

# pca.fit(X_train)

# X_train = pca.transform(X_train)
# X_val = pca.transform(X_val)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)


ours_val = knn.predict_proba(X_val)


print("Metrics on the validation set:")
print_metrics(ours_val, y_val)


# X_test, idx_test = load_test_set()

# # X_test = pca.transform(X_test)

# ours_test = knn.predict_proba(X_test)

# save_results('./results/knn.csv', ours_test, idx_test)
