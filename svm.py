from sklearn.utils import shuffle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from dataloader import load_train_val_set, load_test_set, save_results
from metrics import print_metrics

from utils import seed_everything
import time

SEED = 42
seed_everything(SEED)

X_train, X_val, y_train, y_val = load_train_val_set(
    test_size=0.2, random_state=SEED
)

# X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# X_train = X_train[:10000]
# y_train = y_train[:10000]

# pca = PCA(n_components=50)

# pca.fit(X_train)

# X_train = pca.transform(X_train)
# X_val = pca.transform(X_val)
time_0=time.time()
svm = SVC(
    probability=True,
    random_state=SEED,
    verbose=True
)

svm.fit(X_train, y_train)

time_1=time.time()

ours_val = svm.predict_proba(X_val)

time_2=time.time()
print(f"training time:{time_1-time_0}\n validation time:{time_2-time_1}")

print("Metrics on the validation set:")
print_metrics(ours_val, y_val)


# X_test, idx_test = load_test_set()

# # X_test = pca.transform(X_test)

# ours_test = svm.predict_proba(X_test)

# save_results('./results/svm.csv', ours_test, idx_test)
