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
from scipy.interpolate import interp1d

from utils import export_results, plot_as_emf, get_project_root

root = get_project_root()

logging.basicConfig(filename=root / ".logs/logs_hazard.txt",
                    level=logging.INFO,
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

INV_T = 50


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


def plotting(x, y, x_fit, y_fit, plot_apoe=True):
    """
    Plots the input true hazard
    :param x: dict                                          Original X
    :param y: int                                           Original Y
    :param x_fit: array                                     Fitted X
    :param y_fit: array                                     Fitted Y
    :param plot_apoe: bool                                  Plotting APOE or POE?
    :return: None
    """
    plt.scatter(x, y, label="Seismic hazard")
    plt.loglog(x_fit, y_fit, '--', label="Fitted function")
    plt.grid(True, which="major", ls="--")
    plt.legend(frameon=False, loc='best')
    plt.xlim([1e-3, 100])
    plt.ylim([1e-6, 1])

    plt.xlabel(r"Intensity measure, $s$, [g]")
    if plot_apoe:
        plt.ylabel(r"Annual probability of exceedance, $H(s)$")
    else:
        plt.ylabel(r"Probability of exceedance")


def read_hazard(filename, site_name=None):
    """
    reads provided hazard data and plots them
    Parameters:
    filename: Path
    site_name: str

    Returns
    -------
    data: dict
        im: list
            intensity measure names, e.g., 'PGA', 'SA(0.2)', where 0.2 stands for period in seconds
        s: list[list]
            intensity measure values in [g]
        apoe: list[list]
            Annual probability of exceedance (APoE) or PoE
    """
    extension = filename.suffix

    im = []
    s = []
    apoes = []
    poes = []
    if extension == ".pickle" or extension == ".pkl":
        with open(filename, 'rb') as file:
            [im, s, apoe] = pickle.load(file)
            im = np.array(im)
            s = np.array(s)
            apoes = np.array(apoe)
            poes = 1 - np.exp(-apoes * INV_T)

    elif extension == ".json":
        data = json.load(open(filename))

        data = data[list(data.keys())[0]]

        for key in data:
            im.append(key)
            s.append(data[key]['s'])

            if site_name is None:
                site_name = list(data[key]['sites'].keys())[0]

            apoes.append(data[key]['sites'][site_name]['apoe'])
            poes.append(data[key]['sites'][site_name]['poe'])

    elif extension == ".csv":
        df = pd.read_csv(filename)
        df1 = df[['statistic', 'period']]

        df.drop(['lat', 'lon', 'vs30', 'statistic', 'period'], inplace=True, axis=1)
        df = df.astype(float)

        for index, row in df1.iterrows():
            if row['statistic'] != "mean":
                continue

            im.append(row['period'])
            apoes.append(list(df.iloc[index]))
            s_list = []
            for s_val in df.columns:
                s_list.append(float(re.sub("[^0-9, .]", "", s_val)))
            s.append(s_list)

        poes = 1 - np.exp(-np.array(apoes) * INV_T)

    else:
        logging.error("Extension of hazard file not supported or file format is wrong! Supported options: "
                      ".pickle, .json, .csv")
        raise ValueError("Extension of hazard file not supported or file format is wrong! Supported options: "
                         ".pickle, .json, .csv")

    data = {'im': im, 's': s, 'apoe': apoes, 'poe': poes}

    return data


class HazardFit:
    hazard_fit = None
    s_fit = None
    ITERATOR = np.array([0, 3, 9])

    def __init__(self, output_filename, filename, haz_fit=1, pflag=False, export=True, site_name=None,
                 fit_apoe=True):
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
            3 - least-squares fitting
            4 - linear power law, fitting constrained at two intensity levels
            5 - fitting using Bradley et al. 2008
        pflag: bool
            Plot figures
        export: bool
            Export fitted data or not
        site_name: str
            Site name, required only for hazard.json files, if left None, the first key will be selected
        fit_apoe: bool
        """
        self.output_filename = output_filename
        self.filename = filename
        self.pflag = pflag
        self.haz_fit = haz_fit
        self.export = export
        self.site_name = site_name
        self.fit_apoe = fit_apoe

    def perform_fitting(self):
        """
        Runs the fitting function
        Parameters
        ----------
        Returns
        -------
        hazard_fit: DataFrame
        s_fit: np.array
        """
        data = read_hazard(self.filename, self.site_name)

        if self.haz_fit == 1:
            hazard_fit, s_fit = self.my_fitting(data)
        elif self.haz_fit == 2:
            hazard_fit, s_fit = self.scipy_fitting(data)
        elif self.haz_fit == 3:
            hazard_fit, s_fit = self.leastsq_fitting(data)
        elif self.haz_fit == 4:
            hazard_fit, s_fit = self.power_law(data)
        elif self.haz_fit == 5:
            hazard_fit, s_fit = self.bradley_et_al_2008(data)

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
            Annual Probability of exceedance (APoE) of fitted hazard curves
        s_fit: np.array
            Intensity measure (IM) of fitted hazard curves

        Returns
        -------
        info: dict
            Fitted hazard data including APoE, IMs, Periods and coefficients
        """
        T = np.zeros(len(im))
        for t in range(len(im)):
            try:
                T[t] = im[t].replace('SA(', '').replace(')', '')
            except:
                T[t] = 0.0

        info = {'hazard_fit': hazard_fit, 's_fit': s_fit, 'T': T, 'coefs': coefs}

        return info

    def read_seismic_hazard(self, data):
        im = data['im']
        s = data['s']
        apoe = data['apoe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        return im, s, apoe, s_fit

    def my_fitting_single(self, s, apoe, s_range_to_fit, iterator=None):
        """
        Hazard fitting function by proposed improved solution
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances
        s_range_to_fit: List
            Intensity measures where the fitting will be performed
        iterator: List[int]
            List of 3 integer values where the fitting will be prioritized
        Returns
        -------
        apoe_fit: List
            Fitted APOE
        coef: List[float]
            SAC/FEMA-compatible coefficients, [k0, k1, k2]
        """
        if iterator:
            self.ITERATOR = iterator

        if self.pflag:
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        # Fitting the hazard curves
        coef = np.zeros(3)
        # select iterator depending on where we want to have a better fit
        iterator = self.ITERATOR
        r = np.zeros((len(iterator), len(iterator)))
        a = np.zeros(len(iterator))
        cnt = 0
        for i in iterator:
            r_temp = np.array([1])
            for j in range(1, len(iterator)):
                r_temp = np.append(r_temp, -np.power(np.log(s[i]), j))

            r[cnt] = r_temp
            a[cnt] = apoe[i]
            del r_temp
            cnt += 1

        temp1 = np.log(a)
        temp2 = np.linalg.inv(r).dot(temp1)
        temp2 = temp2.tolist()

        coef[0] = np.exp(temp2[0])
        coef[1] = temp2[1]
        coef[2] = temp2[2]

        apoe_fit = coef[0] * np.exp(-coef[2] * np.power(np.log(s_range_to_fit), 2) -
                                    coef[1] * np.log(s_range_to_fit))

        if self.pflag:
            plotting(s, apoe, s_range_to_fit, apoe_fit, self.fit_apoe)
            plt.show()
            plt.close()

        if self.export:
            info = {
                "x": s,
                "y": apoe,
                "x_fit": s_range_to_fit,
                "y_fit": apoe_fit,
                "apoe": self.fit_apoe,
                "coef": coef,
            }

            export_results(f"{self.output_filename}_shahnazaryan", info, 'json')

        return apoe_fit, coef

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
        im, s, apoe, s_fit = self.read_seismic_hazard(data)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))

        # Fitting the hazard curves
        for k, im_level in enumerate(im):
            coef = np.zeros(3)
            # select iterator depending on where we want to have a better fit
            iterator = self.ITERATOR
            r = np.zeros((len(iterator), len(iterator)))
            a = np.zeros(len(iterator))
            cnt = 0
            for i in iterator:
                r_temp = np.array([1])
                for j in range(1, len(iterator)):
                    r_temp = np.append(r_temp, -np.power(np.log(s[k][i]), j))

                r[cnt] = r_temp
                a[cnt] = apoe[k][i]
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
            hazard_fit[im_level] = H_fit
            coefs[im_level] = coef

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_shahnazaryan", info, 'pickle')

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

        im, s, apoe, s_fit = self.read_seismic_hazard(data)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0, 0, 0])
        sigma = np.array([1.0] * len(s[0]))

        def func(x, a, b, c):
            return a * np.exp(-c * np.power(np.log(x), 2) - b * np.log(x))

        for i, im_level in enumerate(im):
            p, pcov = optimization.curve_fit(func, s[i], apoe[i], x0, sigma)
            H_fit = p[0] * np.exp(-p[2] * np.power(np.log(s_fit), 2) -
                                  p[1] * np.log(s_fit))
            hazard_fit[im_level] = H_fit
            coefs[im_level] = p

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

        im, s, apoe, s_fit = self.read_seismic_hazard(data)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0.01, 0.01, 0.01])

        def func(x, s, a):
            return np.log(a) - np.log(x[0] * np.exp(-x[2] * np.power(np.log(s), 2) - x[1] * np.log(s)))

        for i, im_level in enumerate(im):
            p = optimization.leastsq(func, x0, args=(s[i], apoe[i]), factor=1)[0]
            H_fit = p[0] * np.exp(-p[2] * np.power(np.log(s_fit), 2) -
                                  p[1] * np.log(s_fit))
            hazard_fit[im_level] = H_fit
            coefs[im_level] = p

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_lstsq", info, 'pickle')

        return hazard_fit, s_fit

    def power_law(self, data, dbe=475, mce=10000):
        """
        Performs fitting on a loglinear power law constrained at two intensity levels
        Parameters
        ----------
        data: dict
        dbe: int
            Return period of design basis earthquake level
        mce: int
            Return period of maximum considered earthquake level
        Returns
        -------
        hazard_fit: DataFrame
        s_fit: np.array
        """
        logging.info("[FITTING] Loglinear fitting")

        im, s, apoe, s_fit = self.read_seismic_hazard(data)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k'], columns=list(im))

        for i, im_level in enumerate(im):
            s_range = s[i]
            apoe_range = apoe[i]

            # get constraining intensity levels
            apoe_dbe = 1 / dbe
            apoe_mce = 1 / mce

            interpolation = interp1d(apoe_range, s_range)
            s_dbe = interpolation(apoe_dbe)
            s_mce = interpolation(apoe_mce)

            # Get the fitting coefficients
            k = np.log(apoe_dbe / apoe_mce) / np.log(s_mce / s_dbe)
            k0 = apoe_dbe * s_dbe ** k

            # Fitted APOE
            H_fit = k0 * s_fit ** (-k)

            hazard_fit[im_level] = H_fit
            coefs[im_level] = [k0, k]

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_loglinear", info, 'pickle')

        return hazard_fit, s_fit

    def bradley_et_al_2008(self, data):
        logging.info("[FITTING] Fitting using the analytical approach of Bradley et al. 2008")

        im, s, apoe, s_fit = self.read_seismic_hazard(data)

        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['H_asy', 's_asy', 'alpha'], columns=list(im))
        x0 = np.array([100, 100, 50])

        def func(x, s, a):
            apoe_asy = x[0]
            s_asy = x[1]
            alpha = x[2]
            return np.log(a) - np.log(apoe_asy * np.exp(alpha * (np.log(s / s_asy)) ** -1))

        for i, im_level in enumerate(im):
            p = optimization.leastsq(func, x0, args=(s[i], apoe[i]), factor=100)[0]
            H_fit = p[0] * np.exp(p[2] * (np.log(s_fit / p[1])) ** -1)

            hazard_fit[im_level] = H_fit
            coefs[im_level] = p

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.export:
            export_results(f"{self.output_filename}_bradley", info, 'pickle')

        return hazard_fit, s_fit
