import numpy as np
from sklearn.metrics import roc_curve, auc


#NMF
def auc_per_cancer(X, drug_names, cancer_names, truth):

    aucs = []

    for j, cancer in enumerate(cancer_names):

        y_true = []
        y_score = []

        for i, drug in enumerate(drug_names):
            y_score.append(X[i, j])
            y_true.append((str(drug), str(cancer)) in truth)

        y_true = np.array(y_true)

        if y_true.sum() > 0:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            aucs.append(auc(fpr, tpr))

    return aucs


def recall_at_k(rankings, truth, K):

    recalls = []

    for cancer, df in rankings.items():

        total_pos = sum(
            (str(d), str(cancer)) in truth
            for d in df["drugbank_id"]
        )

        if total_pos == 0:
            continue

        top_k = df.head(K)

        pos_top = sum(
            (str(d), str(cancer)) in truth
            for d in top_k["drugbank_id"]
        )

        recalls.append(pos_top / total_pos)

    return recalls

#Guney

def auc_per_cancer_from_Z(Z_df, truth):
    """
    Calcula a AUC por câncer usando a matriz Z.

    Parâmetros:
    -----------
    Z_df : pd.DataFrame
        Matriz de Z-score (drogas x cânceres)
    truth : set
        Conjunto de pares (drug_id, cancer_id) verdadeiros

    Retorna:
    --------
    aucs : list
        Lista com um valor de AUC por câncer.
    """
    drug_names = list(Z_df.index)
    cancer_names = list(Z_df.columns)

    # Z mais negativo = melhor → inverter sinal para "quanto maior, melhor"
    X = -Z_df.values  # agora z negativo vira positivo

    return auc_per_cancer(X, drug_names, cancer_names, truth)