# =========================================================
# IMPORTS
# =========================================================
from data.loader import load_data
from evaluation.splits import filter_truth_by_available_nodes
from graph.diffusion import build_graph, compute_diffusion_matrix
from features.matrices import *
from models.nmf import test_multiple_k

from evaluation.ground_truth import build_ground_truth
from evaluation.ranking import *
from evaluation.metrics import *

from utils.io import *
from utils.logger import setup_logger
from utils.model_io import save_model

import numpy as np


# =========================================================
# CONFIG
# =========================================================
paths = {
    "ppi": "...",
    "drug_targets": "...",
    "cancer": "...",
    "repo": "..."
}

k_values = [5, 10, 15, 20]
recall_ks = [10, 20, 50]


# =========================================================
# 1. SETUP
# =========================================================
exp_path = create_experiment_folder()
logger = setup_logger(exp_path)

logger.info("Iniciando experimento")


# =========================================================
# 2. LOAD DATA
# =========================================================
ppi, drug_targets, cancer, repoDB = load_data(paths)


# =========================================================
# 3. GRAFO + DIFUSÃO
# =========================================================
G = build_graph(ppi)
K, nodes, node_index = compute_diffusion_matrix(G)


# =========================================================
# 4. MATRIZES A E B
# =========================================================
P_d = build_association_dict(drug_targets, "drugbank_id", "entrez_id")
P_c = build_association_dict(cancer, "diseaseid", "geneid")

A = build_matrix(P_d, nodes, K, node_index)
B = build_matrix(P_c, nodes, K, node_index)

A_bin = binarize(A)
B_bin = binarize(B)

drug_names = A_bin.index.tolist()
cancer_names = B_bin.index.tolist()

A_np = A_bin.to_numpy()
B_np = B_bin.to_numpy()


# =========================================================
# 5. TREINAR MODELOS
# =========================================================
modelos = test_multiple_k(A_np, B_np, k_values)


# =========================================================
# 6. GROUND TRUTH (RepoDB)
# =========================================================
verdade = build_ground_truth(repoDB)

verdade = filter_truth_by_available_nodes(
    verdade,
    drug_names,
    cancer_names
)

logger.info(f"Total pares verdade: {len(verdade)}")


# =========================================================
# 7. LOOP DE AVALIAÇÃO
# =========================================================
resultados = {}

for k, model in modelos.items():

    logger.info(f"Avaliando modelo k={k}")

    # -------------------------------
    # 7.1 MATRIZ DE EFICÁCIA
    # -------------------------------
    X = efficacy_matrix(model)

    # -------------------------------
    # 7.2 RANKING
    # -------------------------------
    rankings = ranking_per_cancer(
        X,
        drug_names,
        cancer_names
    )

    # -------------------------------
    # 7.3 AUC
    # -------------------------------
    auc_medio, auc_por_cancer = auc_per_cancer(
        X,
        drug_names,
        cancer_names,
        verdade
    )

    # -------------------------------
    # 7.4 RECALL
    # -------------------------------
    recall_dict = {}

    for K in recall_ks:
        recall_dict[K] = recall_at_k(
            rankings,
            verdade,
            K
        )

    # -------------------------------
    # 7.5 SALVAR MODELO
    # -------------------------------
    save_model(model, exp_path, f"k_{k}")

    # -------------------------------
    # 7.6 GUARDAR RESULTADOS
    # -------------------------------
    resultados[k] = {
        "auc_medio": auc_medio,
        "auc_por_cancer": auc_por_cancer,
        "recall": recall_dict
    }


# =========================================================
# 8. SALVAR RESULTADOS
# =========================================================
save_json(resultados, f"{exp_path}/results.json")

logger.info("Experimento finalizado")