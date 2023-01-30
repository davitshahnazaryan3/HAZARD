from pathlib import Path

from Hazard.hazard import HazardFit


path = Path.cwd().parents[0]
hazardFileName = path / "sample/Hazard-LAquila-Soil-C.pkl"
# hazardFileName = path / "sample/example1.json"
# hazardFileName = path / "sample/nz_hazard.csv"

outputPath = path / "sample/outputs_old/example2"

h = HazardFit(outputPath, hazardFileName, 1, pflag=True, export=False)
h.perform_fitting()


# TODO
"""
Users input 3 Return periods,

Then select a return period closest to the inputs, and 
get the s based on that.

Default to 3 typical periods, DBE, MCE etc.
"""
