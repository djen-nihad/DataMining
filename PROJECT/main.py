from UI.ui import App
from preprocessing import *
from Experemntations.parametrizeDBSCAN import *


def read_with_preprocess(path):
    df = read_data(path)
    df = df.drop('OC', axis=1)
    for column in df.columns:
        df[column] = replace_missing_values(df_column=df[column], method='median')
    df = delete_duplicate_rows(df)
    df = min_max_normalisation(df)
    array = df.to_numpy()
    X, y = array[:, :-1], array[:, -1].reshape(-1, 1)
    X = X.astype(float)
    y = y.astype(int)
    y = y.ravel()
    return X, y


if __name__ == '__main__':
    path = 'dataset/Dataset1.csv'
    X, y = read_with_preprocess(path)
    eps = 0.38
    min_samples = 20
    appliqueDBSCAN(X, eps=eps, min_samples=min_samples, verbose=True)
