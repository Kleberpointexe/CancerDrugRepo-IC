import numpy as np
import pandas as pd


def build_association_dict(df, key, value):
    return df.groupby(key)[value].apply(list).to_dict()


def perturbation(K, P, node_index):
    idx = [node_index[n] for n in P if n in node_index]
    if not idx:
        return np.zeros(K.shape[0])
    return np.max(K[:, idx], axis=1)


def build_matrix(P_dict, nodes, K, node_index):
    M = pd.DataFrame(index=P_dict.keys(), columns=nodes, dtype=float)

    for key, P in P_dict.items():
        M.loc[key] = perturbation(K, P, node_index)

    return M


def binarize(df, quantile=0.9):
    threshold = df.stack().quantile(quantile)
    return (df >= threshold).astype(int)

