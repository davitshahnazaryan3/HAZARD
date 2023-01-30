from pathlib import Path

from Hazard.hazard import HazardFit


path = Path.cwd().parents[0]
hazardFileName = path / "sample/Hazard-LAquila-Soil-C.pkl"
# hazardFileName = path / "sample/example1.json"
# hazardFileName = path / "sample/nz_hazard.csv"

outputPath = path / "sample/outputs_old/example2"

h = HazardFit(outputPath, hazardFileName, 5, pflag=True, export=False)
h.perform_fitting()
