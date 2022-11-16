"""
Hazard library for fitting the hazard curve and generating a .pickle file
"""
import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimization
import numpy as np
from pathlib import Path
import logging
from utils import export_results

logging.basicConfig(filename="../.logs/logs_hazard.txt",
                    level=logging.DEBUG,
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")


class Hazard:
    def __init__(self, filename, outputDir):
        """
        
        Parameters
        ----------
        filename: str
            Hazard file name
        outputDir: Path
            Hazard directory
        """
        self.filename = filename
        self.outputDir = outputDir

    def read_hazard(self):
        """
        reads fitted hazard data
        Returns
        -------
        coefs: dict
            Coefficients of the fitted hazard curves
        hazard_data: dict
            Fitted hazard curves
        original_hazard: dict
            Original hazard curves
        """
        filename = os.path.basename(self.filename)
        with open(self.outputDir / f"coef_{filename}", 'rb') as file:
            coefs = pickle.load(file)
        with open(self.outputDir / f"fit_{filename}", 'rb') as file:
            hazard_data = pickle.load(file)
        with open(self.filename, 'rb') as file:
            original_hazard = pickle.load(file)

        return coefs, hazard_data, original_hazard


class HazardFit:
    hazard_fit = None
    s_fit = None
    ITERATOR = np.array([0, 3, 9])

    def __init__(self, output_filename, filename, haz_fit=1, pflag=False, export=True, site_name=None):
        """
        init hazard fitting tool
        Parameters
        ----------
        output_filename: Path
            Output filename to export to
        filename: Path
            Hazard file name, e.g., *.pickle or *.pkl or *.csv or *.json
            /sample/ includes examples for each type of file
        haz_fit: int
            Hazard fitting function to use (1, 2, 3)
            1 - improved proposed fitting function (recommended)
            2 - scipy curve_fit
            3 - least squares fitting
        pflag: bool
            Plot figures
        export: bool
            Export fitted data or not
        site_name: str
            Site name, required only for hazard.json files, if left None, the first key will be selected
        """
        self.output_filename = output_filename
        self.filename = filename
        self.pflag = pflag
        self.haz_fit = haz_fit
        self.export = export
        self.site_name = site_name

    def read_hazard(self):
        """
        reads provided hazard data and plots them
        Returns
        -------
        data: dict
            im: list
                intensity measure names, e.g., 'PGA', 'SA(0.2)', where 0.2 stands for period in seconds
            s: list[list]
                intensity measure values in [g]
            poe: list[list]
                Probability of exceedance (PoE) or annual PoE
        """
        extension = self.filename.suffix

        im = []
        s = []
        poes = []
        if extension == ".pickle" or extension == ".pkl":
            with open(self.filename, 'rb') as file:
                [im, s, poe] = pickle.load(file)
                im = np.array(im)
                s = np.array(s)
                poes = np.array(poe)

        elif extension == ".json":
            data = json.load(open(self.filename))

            data = data[list(data.keys())[0]]

            for key in data:
                im.append(key)
                s.append(data[key]['s'])

                if self.site_name is None:
                    self.site_name = list(data[key]['sites'].keys())[0]

                poes.append(data[key]['sites'][self.site_name]['poe'])

        elif extension == ".csv":
            df = pd.read_csv(self.filename)
            df1 = df[['statistic', 'period']]

            df.drop(['lat', 'lon', 'vs30', 'statistic', 'period'], inplace=True, axis=1)
            df = df.astype(float)

            for index, row in df1.iterrows():
                if row['statistic'] != "mean":
                    continue

                im.append(row['period'])
                poes.append(list(df.iloc[index]))
                s_list = []
                for s_val in df.columns:
                    s_list.append(float(re.sub("[^0-9, .]", "", s_val)))
                s.append(s_list)

        else:
            logging.error("Extension of hazard file not supported or file format is wrong! Supported options: "
                          ".pickle, .json, .csv")
            raise ValueError("Extension of hazard file not supported or file format is wrong! Supported options: "
                             ".pickle, .json, .csv")

        data = {'im': im, 's': s, 'poe': poes}

        return data

    @staticmethod
    def plotting(version, data, tag=None, s_fit=None, H_fit=None):
        """
        Plots the input true hazard
        :param version: str                                     Original or Fitted version to plot
        :param data: dict                                       Hazard data
        :param tag: int                                         Record tag
        :param s_fit: array                                     Fitted hazard intensity measures
        :param H_fit: array                                     Fitted hazard probabilities
        :return: None
        """
        if version == "Original":
            for i in range(len(data['im'])):
                plt.loglog(data['s'][i], data['poe'][i])
            plt.grid(True, which="both", ls="--")
            plt.xlim([1e-3, 100])
            plt.ylim([1e-4, 1])
        elif version == "Fitted":
            plt.scatter(data['s'][tag], data['poe'][tag])
            #            self.data['s'][self.tag],self.data['poe'][self.tag],'-',
            plt.loglog(s_fit, H_fit, '--')
            plt.grid(True, which="both", ls="--")
            plt.xlim([1e-3, 100])
            plt.ylim([1e-6, 1])
        else:
            raise ValueError('[EXCEPTION] Wrong version of plotting!')

        plt.xlabel(r"Intensity measure, $s$, [g]")
        plt.ylabel(r"Annual probability of exceedance, $H$")

    def perform_fitting(self):
        """
        Runs the fitting function
        Parameters
        ----------
        Returns
        -------
        hazard_fit: DataFrame
        s_fit: np.array+
        """
        data = self.read_hazard()

        if self.haz_fit == 1:
            hazard_fit, s_fit = self.my_fitting(data)
        elif self.haz_fit == 2:
            hazard_fit, s_fit = self.scipy_fitting(data)
        elif self.haz_fit == 3:
            hazard_fit, s_fit = self.leastsq_fitting(data)
        else:
            logging.error('[EXCEPTION] Wrong fitting function! Must be 1, 2, or 3')
            raise ValueError('[EXCEPTION] Wrong fitting function!')

        return hazard_fit, s_fit

    def generate_fitted_data(self, im, coefs, hazard_fit, s_fit):
        """
        Generates dictionary for exporting hazard data
        Parameters
        ----------
        im: np.array
            Intensity measures
        coefs: DataFrame
            Fitting coefficients
        hazard_fit: DataFrame
            Probability of exceedance (PoE) of fitted hazard curves
        s_fit: np.array
            Intensity measure (IM) of fitted hazard curves

        Returns
        -------
        info: dict
            Fitted hazard data including PoE, IMs, Periods and coefficients
        """
        T = np.zeros(len(im))
        for t in range(len(im)):
            try:
                T[t] = im[t].replace('SA(', '').replace(')', '')
            except:
                T[t] = 0.0

        info = {'hazard_fit': hazard_fit, 's_fit': s_fit, 'T': T, 'coefs': coefs}

        return info

    def my_fitting(self, data):
        """
        Hazard fitting function by proposed improved solution
        Parameters
        ----------
        data: dict
            True hazard data
        Returns
        -------
        hazard_fit: DataFrame
            Fitted Hazard data
        s_fit: array
            Intensity measure of hazard

        """
        logging.info("[FITTING] Hazard Proposed improved solution")
        im = data['im']
        s = data['s']
        poe = data['poe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))

        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)

        # Fitting the hazard curves
        for tag in range(len(im)):
            coef = np.zeros(3)
            # select iterator depending on where we want to have a better fit
            iterator = self.ITERATOR
            r = np.zeros((len(iterator), len(iterator)))
            a = np.zeros(len(iterator))
            cnt = 0
            for i in iterator:
                r_temp = np.array([1])
                for j in range(1, len(iterator)):
                    r_temp = np.append(r_temp, -np.power(np.log(s[tag][i]), j))

                r[cnt] = r_temp
                a[cnt] = poe[tag][i]
                del r_temp
                cnt += 1

            temp1 = np.log(a)
            temp2 = np.linalg.inv(r).dot(temp1)
            temp2 = temp2.tolist()
            coef[0] = np.exp(temp2[0])
            coef[1] = temp2[1]
            coef[2] = temp2[2]

            H_fit = coef[0] * np.exp(-coef[2] * np.power(np.log(s_fit), 2) -
                                     coef[1] * np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = coef

            if self.pflag:
                print(tag, coef, iterator)

                if im[tag] == 'PGA' or im[tag] == 'SA(0.7)':
                    self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)
                    plt.tight_layout()
                    plt.show()

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(self.output_filename, info, 'pickle')

        return hazard_fit, s_fit

    def scipy_fitting(self, data):
        """
        Hazard fitting function by scipy library
        Parameters
        ----------
        data: dict
            True hazard data
        Returns
        -------
        hazard_fit: DataFrame
            Fitted Hazard data
        s_fit: array
            Intensity measure of hazard

        """
        logging.info("[FITTING] Hazard Scipy curve_fit")
        im = data['im']
        s = data['s']
        poe = data['poe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0, 0, 0])
        sigma = np.array([1.0] * len(s[0]))

        def func(x, a, b, c):
            return a * np.exp(-c * np.power(np.log(x), 2) - b * np.log(x))

        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        for tag in range(len(im)):
            p, pcov = optimization.curve_fit(func, s[tag], poe[tag], x0, sigma)
            H_fit = p[0] * np.exp(-p[2] * np.power(np.log(s_fit), 2) -
                                  p[1] * np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = p

            if self.pflag:
                print(tag, p)
                self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)
                plt.show()

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_curve_fit", info, 'pickle')

        return hazard_fit, s_fit

    def leastsq_fitting(self, data):
        """
        Hazard fitting function least squares method
        Parameters
        ----------
        data: dict
            True hazard data
        Returns
        -------
        hazard_fit: DataFrame
            Fitted Hazard data
        s_fit: array
            Intensity measure of hazard
        """
        logging.info("[FITTING] Hazard leastSquare")
        im = data['im']
        s = data['s']
        poe = data['poe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0, 0, 0])

        def func(x, s, a):
            return a - x[0] * np.exp(-x[2] * np.power(np.log(s), 2) - x[1] * np.log(s))

        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        for tag in range(len(im)):
            p = optimization.leastsq(func, x0, args=(s[tag], poe[tag]))[0]
            H_fit = p[0] * np.exp(-p[2] * np.power(np.log(s_fit), 2) -
                                  p[1] * np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = p

            if self.pflag:
                print(tag, p)
                self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)
                plt.show()

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_lstsq", info, 'pickle')

        return hazard_fit, s_fit


if __name__ == "__main__":
    path = Path.cwd().parents[0]
    hazardFileName = path / "sample/example2.csv"
    outputPath = path / "sample/outputs/example2"

    h = HazardFit(outputPath, hazardFileName, 3, pflag=False, export=True)
    h.perform_fitting()
