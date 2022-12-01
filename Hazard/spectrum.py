import json
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from Hazard.utils import get_probability_of_exceedance
from utils import export_results, create_folder

logging.basicConfig(filename="../.logs/logs_spectra.txt",
                    level=logging.DEBUG,
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")


class Spectrum:

    def __init__(self, hazard_path, output_path=None, intensity_measure="sat1"):
        """
        Response spectrum generation based on hazard curves obtained via PSHA
        Parameters
        ----------
        hazard_path: Path
            Path to hazard curves (outputs of PSHA)
        output_path: Path
            Path to export the spectra
        intensity_measure: str
            Intensity measure, sat1 or sa_avg
        """
        self.hazard_curves = {intensity_measure.lower(): {}}
        self.hazard_path = hazard_path
        self.output_path = output_path
        self.intensity_measure = intensity_measure.lower()

        create_folder(self.output_path)

    def get_hazard_filenames(self):
        """
        Gets hazard filenames inside the hazard path
        Returns
        -------
        None
        """
        filenames = []
        if self.intensity_measure == "sat1":
            logging.info("Using Sa(T1) as the intensity measure")

            for file in self.hazard_path.iterdir():
                if file.stem.startswith("hazard_curve-mean-SA"):
                    filenames.append(file)

        elif any(im_name in self.intensity_measure for im_name in ["sa_avg", "avg_sa", "saavg", "avgsa"]):
            logging.info("Using Sa_avg as the intensity measure")

            for file in self.hazard_path.iterdir():
                if file.stem.startswith("hazard_curve-mean-AvgSA"):
                    filenames.append(file)

        else:
            logging.error("Wrong Intensity Measure: Must be sat1 or sa_avg")
            raise ValueError("Wrong Intensity Measure: Must be sat1 or sa_avg")

        self.get_hazard_curve(filenames)

    def get_hazard_curve(self, filenames):
        """
        Retrieves hazard curves obtained from PSHA
        Arguments of the PSHA:
            im: intensity measure name
                s: intensity measure levels, [g]
                lat-lon: latitude - longitude
                    poe: probability of exceedance
                    apoe: annual probability of exceedance

        Parameters
        ----------
        filenames: List[Path]
            Filenames of mean hazard curves, e.g. [*.csv, *.csv, ...]
        Returns
        -------
        None
        """
        key = self.intensity_measure
        for filename in filenames:
            base_filename = filename.stem
            im_name = (base_filename.rsplit('-')[2]).rsplit('_')[0]

            # Load the results in as a dataframe
            df = pd.read_csv(filename, skiprows=1)

            # Get investigation time in years
            with open(filename, 'r') as f:
                temp1 = f.readline().split(',')
                temp2 = list(filter(None, temp1))
                inv_t = float(temp2[5].replace(" investigation_time=", ""))
                start_date = temp2[2].replace(" start_date=", "")

            # Strip out the actual IM values
            iml = list(df.columns.values)[3:]
            iml = [float(i[4:]) for i in iml]

            self.hazard_curves[key][im_name] = {
                'psha_date': start_date,
                's': iml
            }

            # Loop over each site
            for site in df.index:
                lat = df['lat'][site]
                lon = df['lon'][site]
                lat_lon = f"{lat}-{lon}"

                poe = list(df.iloc[site, 3:].values)
                apoe = list(-np.log(1 - np.array(poe)) / inv_t)

                # Update hazard curve information
                self.hazard_curves[key][im_name]["sites"] = {
                    lat_lon: {
                        "poe": poe,
                        "apoe": apoe
                    }
                }

        export_results(self.output_path / self.intensity_measure,
                       self.hazard_curves, "json")
        logging.info("Exporting hazard curves")

    def combine_sa_avg_spectra(self):
        pass

    def derive_response_spectrum(self, return_period=475, hazard_curves_path=None):
        """
        Derive response spectrum based on hazard curves conditioned on a given return period
        Parameters
        ----------
        return_period: int
            Return period in years
        hazard_curves_path: Path
            Path to hazard curves computed, if None, defaults to self.hazard_curves
        Returns
        -------
        Response spectra
        """
        response_spectrum = {}

        if hazard_curves_path is None:
            hazard = self.hazard_curves
        else:
            if hazard_curves_path.is_file():
                # SaT1 cases
                hazard = json.load(open(hazard_curves_path))
                hazard = hazard["sat1"]

            else:
                # Avg_Sa cases
                hazard = {}
                for file in hazard_curves_path.iterdir():
                    if file.stem.startswith("spectrum"):
                        continue

                    data = json.load(open(file))
                    key = list(data.keys())[0]
                    im_name = list(data[key].keys())[0]
                    hazard[key] = data[key][im_name]

        target = get_probability_of_exceedance(period)

        for key in hazard:
            s = hazard[key]["s"]

            period = float(re.sub("[^0-9, .]", "", key))

            for site in hazard[key]["sites"]:
                poe = hazard[key]["sites"][site]["poe"]

                interpolation = interp1d(poe, s)
                im_value = float(interpolation(target))

                if site in response_spectrum:
                    response_spectrum[site]["periods"].append(period)
                    response_spectrum[site]["ims"].append(im_value)
                else:
                    response_spectrum[site] = {
                        "periods": [period],
                        "ims": [im_value],
                    }

        export_results(self.output_path / f"spectrum_{self.intensity_measure}_{return_period}",
                       response_spectrum, "json")

        logging.info(f"Exporting response spectrum based on hazard curves at a return period {return_period} years,"
                     f"Intensity measure: {self.intensity_measure}")

        return response_spectrum


if __name__ == "__main__":
    path = Path("E:/") / "Data-Driven Design"
    outputPath = path / "PSHA/AvgSa/curves"
    hazard_path = path / "PSHA/AvgSa/OQ_Outputs_3.0"
    intensity_measure = "sa_avg_3.0"

    s = Spectrum(hazard_path, outputPath, intensity_measure)
    # s.get_hazard_filenames()

    # sa_avg -------------
    hazard_curves_path = outputPath
    s.derive_response_spectrum(475, hazard_curves_path)

    # sat1 ----------
    # hazard_curves_path = outputPath / "sat1.json"
    # s.derive_response_spectrum(475, hazard_curves_path)
