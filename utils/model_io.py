import numpy as np
import os

def save_model(model, path, name):

    Wd, Wc, H, error_history, it = model

    np.save(os.path.join(path, f"{name}_Wd.npy"), Wd)
    np.save(os.path.join(path, f"{name}_Wc.npy"), Wc)
    np.save(os.path.join(path, f"{name}_H.npy"), H)
    np.save(os.path.join(path, f"{name}_error.npy"), error_history)


def load_model(path, name):

    Wd = np.load(os.path.join(path, f"{name}_Wd.npy"))
    Wc = np.load(os.path.join(path, f"{name}_Wc.npy"))
    H = np.load(os.path.join(path, f"{name}_H.npy"))

    return Wd, Wc, H