import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from multiprocessing import Pool
from functools import partial
import random
from tqdm import tqdm

def precompute_distances(G):
    """Pré-calcula matriz de distâncias entre todos os pares de nós."""
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=bool)
    dist = shortest_path(csgraph=A, directed=False, unweighted=True)
    max_dist = 65535
    dist[np.isinf(dist)] = max_dist
    dist = dist.astype(np.uint16)
    return dist, idx, nodes

def avg_min_distance_fast(A, B, dist, idx, max_dist=65535):
    """Versão vetorizada usando NumPy"""
    if not A or not B:
        return np.inf
    iA = np.array([idx[a] for a in A])
    iB = np.array([idx[b] for b in B])
    # Matriz de distâncias entre os conjuntos (shape len(A) x len(B))
    d = dist[np.ix_(iA, iB)]
    # Menor distância de cada nó em A para qualquer nó em B
    min_dist = np.min(d, axis=1)
    # Filtra nós não conectados (distância = 65535)
    min_dist = min_dist[min_dist < max_dist]
    return np.mean(min_dist) if len(min_dist) > 0 else np.inf

def compute_proximity_pair(args, dist, idx, degree_map, degree_of_node, n_random, base_seed):
    """
    Processa um par droga-câncer: calcula d_obs e Z-score.
    """
    drug, cancer, targets, disease = args
    if not targets or not disease:
        return drug, cancer, np.nan, np.nan

    d_obs = avg_min_distance_fast(targets, disease, dist, idx)
    if np.isinf(d_obs):
        return drug, cancer, d_obs, np.nan

    seed = (base_seed + hash((drug, cancer))) % (2**32)
    rng = random.Random(seed)

    rand_dists = []
    deg_targets = [degree_of_node[t] for t in targets]
    deg_disease = [degree_of_node[d] for d in disease]

    for _ in range(n_random):
        rand_t = [rng.choice(degree_map[deg]) for deg in deg_targets]
        rand_d = [rng.choice(degree_map[deg]) for deg in deg_disease]
        d_rand = avg_min_distance_fast(rand_t, rand_d, dist, idx)
        rand_dists.append(d_rand)

    mean_rand = np.mean(rand_dists)
    std_rand = np.std(rand_dists, ddof=1)
    z = (d_obs - mean_rand) / std_rand if std_rand > 0 else 0.0
    return drug, cancer, d_obs, z

def compute_proximity_matrix_fast(G, P_d, P_c, n_random=100, seed=452456, verbose=True, n_workers=None):
    """
    Calcula matriz de proximidade (Z e d) otimizada.

    Parâmetros:
    -----------
    G : networkx.Graph
        Grafo PPI
    P_d : dict
        {drug_id: [gene_id, ...]}
    P_c : dict
        {cancer_id: [gene_id, ...]}
    n_random : int
        Número de randomizações (default 1000)
    seed : int
        Semente global
    verbose : bool
        Mostrar barra de progresso
    n_workers : int or None
        Número de processos paralelos (None = usa CPU count)

    Retorna:
    --------
    Z_df : pd.DataFrame
        Matriz de Z-scores
    D_df : pd.DataFrame
        Matriz de distâncias brutas
    """
    print("Pré-calculando matriz de distâncias...")
    dist, idx, nodes = precompute_distances(G)
    print(f"Matriz de distâncias criada: {dist.shape}")

    print("Construindo mapeamento de graus...")
    degree_map = {}
    degree_of_node = {}
    for node, deg in G.degree():
        degree_map.setdefault(deg, []).append(node)
        degree_of_node[node] = deg

    node_set = set(nodes)
    drug_targets = {d: [g for g in genes if g in node_set] for d, genes in P_d.items()}
    cancer_genes = {c: [g for g in genes if g in node_set] for c, genes in P_c.items()}

    pairs = []
    for d, targs in drug_targets.items():
        if not targs:
            continue
        for c, dis in cancer_genes.items():
            if dis:
                pairs.append((d, c, targs, dis))

    print(f"Total de pares a processar: {len(pairs)}")

    # Cria partial com todos os argumentos fixos
    func_partial = partial(compute_proximity_pair,
                           dist=dist,
                           idx=idx,
                           degree_map=degree_map,
                           degree_of_node=degree_of_node,
                           n_random=n_random,
                           base_seed=seed)

    if n_workers is None:
        n_workers = 4  # Ajuste conforme sua máquina

    if verbose:
        print(f"Iniciando processamento com {n_workers} workers...")

    results = []
    with Pool(processes=n_workers) as pool:
        iterator = pool.imap(func_partial, pairs)
        if verbose:
            iterator = tqdm(iterator, total=len(pairs), desc="Processando pares")
        results = list(iterator)

    drug_ids = list(drug_targets.keys())
    cancer_ids = list(cancer_genes.keys())
    Z_df = pd.DataFrame(index=drug_ids, columns=cancer_ids, dtype=float)
    D_df = pd.DataFrame(index=drug_ids, columns=cancer_ids, dtype=float)

    for drug, cancer, d_obs, z in results:
        D_df.loc[drug, cancer] = d_obs
        Z_df.loc[drug, cancer] = z

    return Z_df, D_df