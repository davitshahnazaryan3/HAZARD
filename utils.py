import numpy as np
import pickle
import json


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
