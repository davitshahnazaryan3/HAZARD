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

from Hazard.parse_usgs_hazard import usgs_hazard
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
    :return: fig, ax
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
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

    plt.show()
    plt.close()
    return fig, ax


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

    elif extension == ".json" or extension == ".txt":
        if extension == ".txt":
            data = usgs_hazard(filename)
        else:
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


def read_seismic_hazard(data):
    im = data['im']
    s = data['s']
    apoe = data['apoe']
    s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
    return im, s, apoe, s_fit


def error_function(x, s, a):
    return np.log(a) - np.log(x[0] * np.exp(-x[2] * np.power(np.log(s), 2) - x[1] * np.log(s)))


class HazardFit:
    ITERATOR = [0, 3, 9]
    s_range_to_fit = None

    def __init__(self, output_filename, filename, haz_fit=1, pflag=False, export=True, site_name=None,
                 fit_apoe=True, im=None, apoe=None):
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
            Hazard fitting function to use (1, 2, 3, 4)
            1 - improved proposed fitting function (recommended)
            2 - least-squares fitting
            3 - linear power law, fitting constrained at two intensity levels
            4 - fitting using Bradley et al. 2008
        pflag: bool
            Plot figures
        export: bool
            Export fitted data or not
        site_name: str
            Site name, required only for hazard.json files, if left None, the first key will be selected
        fit_apoe: bool
        im: list
            IM range for hazard, if provided, will ignore filename
        apoe: list
            APOE range for hazard, if provided, will ignore filename
        """
        self.output_filename = output_filename
        self.filename = filename
        self.pflag = pflag
        self.haz_fit = haz_fit
        self.export = export
        self.site_name = site_name
        self.fit_apoe = fit_apoe
        self.im = im
        self.apoe = apoe

    def remove_zeros(self, x, y):
        """
        Gets rid of trailing zeros to avoid bias in fitting
        Returns
        -------
        x: List
        y: List
        """
        x = np.array(x)
        y = np.array(y)

        x = x[y > 0]
        y = y[y > 0]
        return x, y

    def perform_fitting(self, im_level=0, iterator=None, dbe=475, mce=10000):
        """
        Runs the fitting function
        Parameters
        im_level: int
            Intensity measure level index
        iterator: List[int]
            List of 3 integer values where the fitting will be prioritized
            Necessary only for haz_fit=1
        dbe: float
            First return period
            For haz_fit=3 only
        mce: float
            Second return period
            For haz_fit=3 only
        ----------
        Returns
        -------
        hazard_fit: DataFrame
        s_fit: np.array
        """
        # init
        if self.im is None:
            data = read_hazard(self.filename, self.site_name)

            im = data['im'][im_level]
            s = data['s'][im_level]

            if self.fit_apoe:
                y = data['apoe'][im_level]
            else:
                y = data['poe'][im_level]

            print("Hazard at IMs:")
            print(data["im"])
            print(f"Number of available IM levels: {len(data['im'])}")
            print(f"Number of hazard points: {len(s)}")
            print(f"Using IM of {im}")

            logging.info(f"[FITTING] Hazard - method: {self.haz_fit} - IML: {im}")
        else:
            im = ""

            s = self.im
            y = self.apoe

        # Range of IM values, where fitting will be performed
        self.s_range_to_fit = np.linspace(min(s), max(s), 1000)

        # Get rid of trailing zeros
        s, y = self.remove_zeros(s, y)

        if self.haz_fit == 1:
            out = self.my_fitting(s, y, iterator=iterator)
        elif self.haz_fit == 2:
            out = self.leastsq_fitting(s, y)
        elif self.haz_fit == 3:
            out = self.power_law(s, y, dbe, mce)
        elif self.haz_fit == 4:
            out = self.bradley_et_al_2008(s, y)

        else:
            logging.error('[EXCEPTION] Wrong fitting function! Must be 1, 2, 3, or 4')
            raise ValueError('[EXCEPTION] Wrong fitting function!')

        if self.export:
            print("Results exported to:")
            print(f"{self.output_filename}/{self.haz_fit}-{im}")
            export_results(f"{self.output_filename}/{self.haz_fit}-{im}", out, 'json')

        return out

    def _into_json_serializable(self, s, apoe, apoe_fit, coef):
        info = {
            "x": list(s),
            "y": list(apoe),
            "x_fit": list(self.s_range_to_fit),
            "y_fit": list(apoe_fit),
            "apoe": self.fit_apoe,
            "coef": list(coef),
        }
        return info

    def second_order_law(self, coef):
        apoe_fit = coef[0] * np.exp(-coef[2] * np.power(np.log(self.s_range_to_fit), 2) -
                                    coef[1] * np.log(self.s_range_to_fit))
        return apoe_fit

    def first_order_law(self, coef):
        return coef[0] * self.s_range_to_fit ** (-coef[1])

    def my_fitting(self, s, apoe, iterator=None):
        """
        Hazard fitting function by proposed improved solution
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances (or POE)
        iterator: List[int]
            List of 3 integer values where the fitting will be prioritized
        Returns
        -------
        dict:
            x: List
                Original IM range
            y: List
                Original APOE or POE
            x_fit: List
                Fitted IM range
            y_fit: List
                Fitted APOE or POE
            apoe: bool
                APOE or POE?
            coef: List[float]
                SAC/FEMA-compatible coefficients, [k0, k1, k2]
            iterator: List[int]
        """

        if iterator:
            self.ITERATOR = iterator

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

        apoe_fit = self.second_order_law(coef)

        if self.pflag:
            plotting(s, apoe, self.s_range_to_fit, apoe_fit, self.fit_apoe)

        info = self._into_json_serializable(s, apoe, apoe_fit, coef)
        info['iterator'] = self.ITERATOR

        return info

    def scipy_fit(self, s, apoe):
        """
        In case LeastSq fails - Hazard fitting function least squares method
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances (or POE)
        Returns
        -------
        dict:
            x: List
                Original IM range
            y: List
                Original APOE or POE
            x_fit: List
                Fitted IM range
            y_fit: List
                Fitted APOE or POE
            apoe: bool
                APOE or POE?
            coef: List[float]
                SAC/FEMA-compatible coefficients, [k0, k1, k2]
        """
        x0 = np.array([0, 0, 0])
        sigma = np.array([1.0] * len(s))

        def func(x, a, b, c):
            return a * np.exp(-c * np.power(np.log(x), 2) - b * np.log(x))

        p, pcov = optimization.curve_fit(func, s, apoe, x0, sigma)
        apoe_fit = p[0] * np.exp(-p[2] * np.power(np.log(self.s_range_to_fit), 2) -
                              p[1] * np.log(self.s_range_to_fit))

        if self.pflag:
            plotting(s, apoe, self.s_range_to_fit, apoe_fit, self.fit_apoe)

        info = self._into_json_serializable(s, apoe, apoe_fit, p)

        return info

    def leastsq_fitting(self, s, apoe):
        """
        Hazard fitting function least squares method
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances (or POE)
        Returns
        -------
        dict:
            x: List
                Original IM range
            y: List
                Original APOE or POE
            x_fit: List
                Fitted IM range
            y_fit: List
                Fitted APOE or POE
            apoe: bool
                APOE or POE?
            coef: List[float]
                SAC/FEMA-compatible coefficients, [k0, k1, k2]
        """
        x0 = np.array([0.1, 0.1, 0.1])
        p = optimization.leastsq(error_function, x0, args=(s, apoe), factor=1)[0]
        apoe_fit = self.second_order_law(p)

        if self.pflag:
            plotting(s, apoe, self.s_range_to_fit, apoe_fit, self.fit_apoe)

        info = self._into_json_serializable(s, apoe, apoe_fit, p)

        return info

    def power_law(self, s, apoe, dbe=465, mce=10000):
        """
        Performs fitting on a loglinear power law constrained at two intensity levels (IMLs)
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances (or POE)
        dbe: int
            Return period of first constrained IML
        mce: int
            Return period of second constrained IML
        Returns
        -------
        dict:
            x: List
                Original IM range
            y: List
                Original APOE or POE
            x_fit: List
                Fitted IM range
            y_fit: List
                Fitted APOE or POE
            apoe: bool
                APOE or POE?
            coef: List[float]
                SAC/FEMA-compatible coefficients, [k0, k]
        """
        # get constraining intensity levels
        apoe_dbe = 1 / dbe
        apoe_mce = 1 / mce

        interpolation = interp1d(apoe, s)
        s_dbe = interpolation(apoe_dbe)
        s_mce = interpolation(apoe_mce)

        # Get the fitting coefficients
        k = np.log(apoe_dbe / apoe_mce) / np.log(s_mce / s_dbe)
        k0 = apoe_dbe * s_dbe ** k

        # Fitted APOE
        coef = [k0, k]
        apoe_fit = self.first_order_law(coef)

        if self.pflag:
            plotting(s, apoe, self.s_range_to_fit, apoe_fit, self.fit_apoe)

        info = self._into_json_serializable(s, apoe, apoe_fit, coef)

        return info

    def bradley_et_al_2008(self, s, apoe):
        """
        Parameters
        ----------
        s: List
            Intensity measures
        apoe: List
            Annual probability of exceedances (or POE)

        Returns
        -------
        dict:
            x: List
                Original IM range
            y: List
                Original APOE or POE
            x_fit: List
                Fitted IM range
            y_fit: List
                Fitted APOE or POE
            apoe: bool
                APOE or POE?
            coef: List
                Fitting coefficients [H_asy, s_asy, alpha]
        """
        def func(x, s, a):
            apoe_asy = x[0]
            s_asy = x[1]
            alpha = x[2]
            return np.log(a) - np.log(apoe_asy * np.exp(alpha * (np.log(s / s_asy)) ** -1))

        x0 = np.array([100, 100, 50])

        p = optimization.leastsq(func, x0, args=(s, apoe), factor=100)[0]
        apoe_fit = p[0] * np.exp(p[2] * (np.log(self.s_range_to_fit / p[1])) ** -1)

        if self.pflag:
            plotting(s, apoe, self.s_range_to_fit, apoe_fit, self.fit_apoe)

        info = self._into_json_serializable(s, apoe, apoe_fit, p)

        return info
