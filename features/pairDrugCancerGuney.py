def get_drug_cancer_pairs(P_d, P_c):
    """
    Retorna:
    --------
    drug_ids : list
        Lista de IDs de drogas
    cancer_ids : list
        Lista de IDs de cânceres
    pairs : list of tuple
        Todas as combinações (drug_id, cancer_id)
    """
    drug_ids = list(P_d.keys())
    cancer_ids = list(P_c.keys())
    pairs = [(d, c) for d in drug_ids for c in cancer_ids]
    return drug_ids, cancer_ids, pairs