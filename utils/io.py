import os
import json
import numpy as np
import pandas as pd
import pickle


def create_experiment_folder(base="results"):
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, timestamp)

    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_numpy(array, path):
    np.save(path, array)


def save_dataframe(df, path):
    df.to_csv(path, index=True)
    
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