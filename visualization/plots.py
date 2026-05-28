import matplotlib.pyplot as plt
import numpy as np
import os


# =========================================================
# CRIAR PASTA
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# 1. AUC BOXPLOT
# =========================================================
def plot_auc_boxplot(resultados_auc, save_path):

    ensure_dir(save_path)

    k_values = sorted(resultados_auc.keys())
    data = [resultados_auc[k] for k in k_values]

    plt.figure(figsize=(10, 6))

    plt.boxplot(data, labels=k_values)
    plt.axhline(0.5, linestyle='--')

    plt.xlabel("Dimensão latente (k)")
    plt.ylabel("AUC por câncer")
    plt.title("Distribuição de AUC por câncer")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "auc_boxplot.png"))
    plt.close()


# =========================================================
# 2. RECALL@K BOXPLOTS
# =========================================================
def plot_recall_boxplots(resultados_recall, save_path, k_list=[10, 20, 50, 150]):

    ensure_dir(save_path)

    k_latentes = sorted(resultados_recall.keys())

    for K in k_list:

        plt.figure(figsize=(12, 6))

        data = [resultados_recall[k][K] for k in k_latentes]

        box = plt.boxplot(
            data,
            labels=k_latentes,
            patch_artist=True,
            medianprops=dict(linewidth=2)
        )

        # transparência das caixas
        for patch in box['boxes']:
            patch.set_alpha(0.5)

        # jitter (pontos individuais)
        for i, k in enumerate(k_latentes):
            y = resultados_recall[k][K]
            x = np.random.normal(i + 1, 0.05, size=len(y))
            plt.plot(x, y, 'o', alpha=0.35)

        # linhas de referência
        plt.axhline(0.0, linestyle='--', linewidth=1)
        plt.axhline(0.5, linestyle='--', linewidth=1)

        plt.ylim(-0.02, 1.02)

        plt.xlabel("Dimensão latente (k)")
        plt.ylabel(f"Recall@{K}")
        plt.title(f"Distribuição de Recall@{K} por câncer")

        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"recall_at_{K}.png"))
        plt.close()


# =========================================================
# 3. CONVERGÊNCIA DO MODELO
# =========================================================
def plot_convergence(modelos, save_path):

    ensure_dir(save_path)

    plt.figure(figsize=(12, 7))

    for k in modelos:

        # modelo retorna: Wd, Wc, H, hist, it
        Wd, Wc, H, hist, it = modelos[k]

        plt.plot(hist, label=f'k={k} ({it} iterações)', linewidth=2)

    plt.title('Convergência por dimensão latente (k)')
    plt.xlabel('Iterações')
    plt.ylabel('Erro de reconstrução')

    plt.yscale('log')  # essencial pra visualizar bem

    plt.grid(True, which="both", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "convergence.png"))
    plt.close()