from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from utils import seed_everything

from numpy.typing import NDArray


def load_dataset() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    df = pd.read_csv('./data/train.csv')

    X_ret = []
    y_ret = []

    for i in range(len(df)):
        item = df.loc[i, 'heartbeat_signals']
        label = df.loc[i, 'label']
        idx = df.loc[i, 'id']

        item = str(item)
        item = [float(i) for i in item.split(',')]

        item = np.array(item)
        idx = int(label)  # type: ignore

        X_ret.append(item)
        y_ret.append(idx)

    X_ret = np.stack(X_ret)
    y_ret = np.stack(y_ret)

    return X_ret, y_ret


def load_train_val_set(**kwargs) -> tuple[
    NDArray[np.float64], NDArray[np.float64],
    NDArray[np.int64], NDArray[np.int64]
]:
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, **kwargs)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    SEED = 42
    seed_everything(SEED)

    X_train, X_val, y_train, y_val = load_train_val_set(
        test_size=0.2, random_state=SEED
    )

    print("Training set size:")
    print(X_train.shape)
    print(y_train.shape)

    print("Validation set size:")
    print(X_val.shape)
    print(y_val.shape)
