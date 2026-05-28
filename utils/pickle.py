import pickle
import os

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
def save_experiment(exp_path, kernel, A, B, modelos, resultados):
    os.makedirs(exp_path, exist_ok=True)

    save_pickle(kernel,     f"{exp_path}/kernel.pkl")
    save_pickle(A,          f"{exp_path}/A.pkl")
    save_pickle(B,          f"{exp_path}/B.pkl")
    save_pickle(modelos,    f"{exp_path}/modelos.pkl")
    save_pickle(resultados, f"{exp_path}/resultados.pkl")


def load_experiment(exp_path):
    return {
        "kernel":     load_pickle(f"{exp_path}/kernel.pkl"),
        "A":          load_pickle(f"{exp_path}/A.pkl"),
        "B":          load_pickle(f"{exp_path}/B.pkl"),
        "modelos":    load_pickle(f"{exp_path}/modelos.pkl"),
        "resultados": load_pickle(f"{exp_path}/resultados.pkl"),
    }