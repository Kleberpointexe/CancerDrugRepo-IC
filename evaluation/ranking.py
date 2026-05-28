import pandas as pd

def efficacy_matrix(model):
    Wd, Wc, H, _, _ = model
    return Wd @ Wc.T


def ranking_per_cancer(X, drug_names, cancer_names):

    rankings = {}

    for j, cancer in enumerate(cancer_names):
        df = pd.DataFrame({
            "drugbank_id": drug_names,
            "score": X[:, j]
        }).sort_values("score", ascending=False)

        rankings[cancer] = df.reset_index(drop=True)

    return rankings

def build_ranking_dict(Z_df, proximal_threshold=None):
    """
    Transforma a matriz Z num dicionário de rankings por câncer.

    Parâmetros:
    -----------
    Z_df : pd.DataFrame
        Matriz de z-score (drogas x cânceres)
    proximal_threshold : float or None
        Se fornecido, mantém apenas drogas com Z < threshold
        (ex: threshold=0 → apenas tratamentos "causativos")

    Retorna:
    --------
    rankings : dict
        {cancer_id: pd.DataFrame com colunas ["drugbank_id", "Z"]}
        DataFrames ordenados por Z ASCENDENTE (menor Z primeiro).
    """
    rankings = {}
    for cancer in Z_df.columns:
        # Ordenar por Z (menor = mais proximal)
        sorted_drugs = Z_df[cancer].sort_values(ascending=True)

        if proximal_threshold is not None:
            sorted_drugs = sorted_drugs[sorted_drugs < proximal_threshold]

        # Criar DataFrame com as duas colunas que a recall_at_k espera
        df_cancer = pd.DataFrame({
            "drugbank_id": sorted_drugs.index,
            "Z": sorted_drugs.values
        })
        rankings[cancer] = df_cancer.reset_index(drop=True)
    return rankings