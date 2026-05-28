import numpy as np
import networkx as nx

def build_graph(ppi):
    G = nx.from_pandas_edgelist(
        ppi,
        source='proteinA_entrezid',
        target='proteinB_entrezid',
        edge_attr='databases',
        create_using=nx.Graph()
    )
    return G


def compute_diffusion_matrix(G, a=2, p=2):
    nodes = sorted(G.nodes())
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()

    I = np.eye(L.shape[0])
    K = np.linalg.matrix_power(a * I - L, p)

    node_index = {n: i for i, n in enumerate(nodes)}

    return K, nodes, node_index