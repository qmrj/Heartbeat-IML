from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from typing import Optional
from numpy.typing import NDArray


def load_dataset(path: str) -> tuple[
    NDArray[np.float64], Optional[NDArray[np.int64]], NDArray[np.int64]
]:
    df = pd.read_csv(path)

    idx_ret = df['id'].to_numpy(dtype=np.int64)

    X_ret = df['heartbeat_signals'].to_numpy()
    X_ret = np.genfromtxt(X_ret, delimiter=',', dtype=np.float64)

    if 'label' in df.keys():
        y_ret = df['label'].to_numpy(dtype=np.int64)
    else:
        y_ret = None

    return X_ret, y_ret, idx_ret


def load_train_val_set(**kwargs) -> tuple[
    NDArray[np.float64], NDArray[np.float64],
    NDArray[np.int64], NDArray[np.int64]
]:
    X, y, _ = load_dataset('./data/train.csv')
    assert y is not None
    X_train, X_val, y_train, y_val = train_test_split(X, y, **kwargs)
    return X_train, X_val, y_train, y_val


def load_test_set() -> tuple[
    NDArray[np.float64], NDArray[np.int64],
]:
    X, y, idx = load_dataset('./data/testA.csv')
    assert y is None
    return X, idx


if __name__ == '__main__':
    from utils import seed_everything

    SEED = 42
    seed_everything(SEED)

    X_train, X_val, y_train, y_val = load_train_val_set(
        test_size=0.2, random_state=SEED
    )

    X_test, idx_test = load_test_set()

    print("Training set size:")
    print(X_train.shape)
    print(y_train.shape)

    print("Validation set size:")
    print(X_val.shape)
    print(y_val.shape)

    print("Test set size:")
    print(X_test.shape)
