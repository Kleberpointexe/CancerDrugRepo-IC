import pandas as pd

def load_data(paths):
    ppi = pd.read_csv(paths["ppi"])
    drug_targets = pd.read_csv(paths["drug_targets"])
    cancer = pd.read_csv(paths["cancer"])
    repo = pd.read_csv(paths["repo"])

    return ppi, drug_targets, cancer, repo