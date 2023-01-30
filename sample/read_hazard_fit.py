import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Hazard.hazard import read_hazard, plotting

path = Path.cwd()

# filename = path / "Hazard-LAquila-Soil-C.pkl"
# filename_fit = path / "outputs_old/Hazard-LAquila-Soil-C_fitted.pickle"
filename = path / "example1.json"
filename_fit = path / "outputs_old/example1.pickle"

hazard = read_hazard(filename)

with open(filename_fit, 'rb') as file:
    fitted = pickle.load(file)

# h_fit = fitted['hazard_fit']["PGA"]
h_fit = fitted['hazard_fit']["SA(0.01)"]
s_fit = fitted["s_fit"]


plotting("Fitted", hazard, 0, s_fit, h_fit, True)
plt.show()
