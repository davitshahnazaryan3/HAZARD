import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats
import pickle
import stat
import json
import errno
import os


def export_results(filepath, data, filetype):
    """
    Store results in the database
    :param filepath: str                            Filepath, e.g. "directory/name"
    :param data:                                    Data to be stored
    :param filetype: str                            Filetype, e.g. npy, json, pkl, csv
    :return: None
    """
    if filetype == "npy":
        np.save(f"{filepath}.npy", data)
    elif filetype == "pkl" or filetype == "pickle":
        with open(f"{filepath}.pickle", 'wb') as handle:
            pickle.dump(data, handle)
    elif filetype == "json":
        with open(f"{filepath}.json", "w") as json_file:
            json.dump(data, json_file)
    elif filetype == "csv":
        data.to_csv(f"{filepath}.csv", index=False)


def create_dir(dir_path):
    """
    Details
    -------
    Creates a clean directory by deleting it if it exists.

    Parameters
    ----------
    dir_path : Path
        name of directory to create.

    None.
    """

    def handle_remove_read_only(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            func(path)
        else:
            raise Warning("Path is being used by at the moment.",
                          "It cannot be recreated.")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=False, onerror=handle_remove_read_only)
    os.makedirs(dir_path)


def get_project_root() -> Path:
    return Path(__file__).parent


def create_folder(directory):
    """
    creates directory if it does not exists
    :param directory: str                   Directory to be created
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def get_range(val, min_factor=0.2, max_factor=2.0, n=10, decimals=4):
    min_val = val * min_factor
    max_val = val * max_factor

    step = (max_val - min_val) / (n - 1)
    arr = [round(min_val, 4)]
    for i in range(1, n):
        arr.append(round(arr[i-1] + step, decimals))

    return arr


def plot_as_emf(figure, **kwargs):
    """
    Saves figure as .emf
    :param figure: fig handle
    :param kwargs: filepath: str                File name, e.g. '*\filename'
    :return: None
    """
    inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//bin//inkscape.exe")
    filepath = kwargs.get('filename', None)
    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)
        svg_filepath = os.path.join(path, filename + '.svg')
        emf_filepath = os.path.join(path, filename + '.emf')
        figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
        # os.remove(svg_filepath)


def mlefit(theta, num_recs, num_collapse, IM):
    """
    Performs a lognormal CDF fit to fragility data points based on maximum likelihood method
    :param theta: float             Medians and standard deviations of the function
    :param num_recs: int            Number of records
    :param num_collapse: int        Number of collapses
    :param IM: list                 Intensity measures
    :return: float
    """
    p = stats.norm.cdf(np.log(IM), loc=np.log(theta[0]), scale=theta[1])
    likelihood = stats.binom.pmf(num_collapse, num_recs, p)
    likelihood[likelihood == 0] = 1e-290
    loglik = -sum(np.log10(likelihood))
    return loglik


def read_pickle_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
