import pickle
from pathlib import Path
import numpy as np


class Spectrum:
    def __init__(self, output_path, intensity_measure="sat1"):
        """
        Derive spectrum from hazard curves
        :param output_path: Path
        :param intensity_measure: str                   sat1 or sa_avg
        """
        self.output_path = output_path
        self.im = intensity_measure

    @staticmethod
    def read_hazard(hazard_path):
        with open(hazard_path, 'rb') as file:
            [im, s, apoe] = pickle.load(file)
            im = np.array(im)
            s = np.array(s)
            apoe = np.array(apoe)
        data = {'im': im, 's': s, 'apoe': apoe}

        return data


if __name__ == "__main__":
    path = Path.cwd().parents[0]
    hazardFileName = path / "sample/Hazard-LAquila-Soil-C.pkl"
    outputPath = path / "sample"

    s = Spectrum(outputPath)
    hazard = s.read_hazard(hazardFileName)
    print(hazard['im'])
