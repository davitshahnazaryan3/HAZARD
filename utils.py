import shutil
import numpy as np
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
