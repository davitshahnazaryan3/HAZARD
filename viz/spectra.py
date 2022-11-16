import matplotlib.pyplot as plt
from pathlib import Path
import json

path = Path("E:/") / "Data-Driven Design/PSHA"
path1 = path / "AvgSa/curves/spectrum_sa_avg_3.0_475.json"
path2 = path / "SaT/curves/sat1_spectrum_475.json"

spectrum_avgsa = json.load(open(path1))
spectrum_sat = json.load(open(path2))
site = list(spectrum_sat.keys())[0]

fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

plt.plot(spectrum_avgsa[site]["periods"],
         spectrum_avgsa[site]["ims"], color="r", label=r"$Sa_{avg}$")
plt.plot(spectrum_sat[site]["periods"],
         spectrum_sat[site]["ims"], color="b", ls="--", label=r"$Sa(T_1)$")

plt.grid(True, which="both", ls="--")
plt.xlim([0, 3])
plt.ylim([0, 2.5])
plt.legend(frameon=False, loc='best')

plt.xlabel(r"Period, $T$ [s]")
plt.ylabel(r"Intensity measure, $s$ [g]")

plt.show()
