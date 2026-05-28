def build_ground_truth(repo_df, status_filter="approved"):
    """
    Filtra o RepoDB e retorna o conjunto de pares (drug, cancer)
    considerados verdadeiros.
    """

    repo_filtered = repo_df[
        repo_df["status"].str.lower() == status_filter
    ].copy()

    truth = set(
        zip(
            repo_filtered["drug_id"].astype(str),
            repo_filtered["ind_id"].astype(str)
        )
    )

    return truth
def analyze_overlap(repo_df, drug_names, cancer_names):

    drug_model = set(map(str, drug_names))
    cancer_model = set(map(str, cancer_names))

    drug_repo = set(repo_df["drug_id"].astype(str))
    cancer_repo = set(repo_df["ind_id"].astype(str))

    drugs_inter = drug_model & drug_repo
    cancer_inter = cancer_model & cancer_repo

    print("Drogas no modelo:", len(drug_model))
    print("Drogas no RepoDB:", len(drgs_inter := drugs_inter))

    print("Cânceres no modelo:", len(cancer_model))
    print("Cânceres no RepoDB:", len(cancer_inter))

    return drugs_inter, cancer_inter