import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from Hazard.hazard import read_hazard
from exceedanceRates.edp_exceedance import compute_exceedance_rates
from utils import plot_as_emf, read_pickle_file
from viz.plot_styles import *


def get_model_label(name):
    if name == 1:
        label = "Proposal"
    elif name == 3:
        label = "Log-linear"
    elif name == 2:
        label = "Least-squares"
    elif name == 4:
        label = "Bradley et al., 2007"
    else:
        label = None
    return label


def get_hazard_data(im_level, hazard_filename, model_filename):
    # Read the hazard and the fitting model
    hazard = read_hazard(hazard_filename)
    model = read_pickle_file(model_filename)

    # Base model name
    method = model_filename.stem.split("_")[-1]
    label = get_model_label(method)

    idx = np.where(hazard["im"] == im_level)[0][0]

    # Hazard from PSHA
    s = hazard["s"][idx]
    apoe = hazard["apoe"][idx]

    # Fitted model
    s_fit = model["s_fit"]
    apoe_fit = model["hazard_fit"][im_level].values
    coefs = model["coefs"][im_level].values

    return s, apoe, s_fit, apoe_fit, label, coefs


def get_return_periods(out):
    rp = []
    for period in out.keys():
        rp.append(int(period))
    rp.sort()

    return rp


def get_peak_response(out, nst, factor=1.0):
    rps = get_return_periods(out)

    edp = {"drift": {"1": [], "2": []}, "acc": []}

    for rp in rps:
        rp = str(rp)

        # Initialize critical EDPs
        mpsd_x, mpsd_y, mpfa = None, None, None
        for st in range(nst + 1):
            if st != nst:
                candidate_x = np.array(out[rp]["1"]["drift"][str(st)]) * 100
                candidate_y = np.array(out[rp]["2"]["drift"][str(st)]) * 100
                if mpsd_x is not None:
                    mpsd_x = np.maximum(mpsd_x, candidate_x)
                    mpsd_y = np.maximum(mpsd_y, candidate_y)
                else:
                    mpsd_x = candidate_x
                    mpsd_y = candidate_y

            candidate = np.maximum(np.array(out[rp]["1"]["acc"][str(st)]),
                                   np.array(out[rp]["2"]["acc"][str(st)])) * factor
            if mpfa is not None:
                mpfa = np.maximum(mpfa, candidate)
            else:
                mpfa = candidate

        # Append MPSD and MPFA
        edp["drift"]["1"].append(mpsd_x)
        edp["drift"]["2"].append(mpsd_y)
        edp["acc"].append(mpfa)

    return edp


class HazardPlots:
    def __init__(self, fit_directory):
        self.fit_directory = fit_directory

    def plot_hazards(self):

        # Plotting
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        styles = {}

        im_cnt = 0
        cnt = 0
        # labels = ["Targeting high intensity", "Targeting low intensity"]
        for path in self.fit_directory.iterdir():
            file = path.stem
            method = int(file[0])
            im_level = file[2:]
            name = get_model_label(method)

            # label = labels[cnt]
            # cnt += 1

            if method not in styles:
                styles[method] = name
                label = styles[method]
            else:
                label = None

            if im_level not in styles:
                styles[im_level] = linestyles[im_cnt]
                im_cnt += 1

            data = json.load(open(path))
            x = data["x"]
            y = data["y"]
            x_fit = data["x_fit"]
            y_fit = data["y_fit"]

            plt.loglog(x_fit, y_fit, label=label, color=alt_color_grid[method-1], ls=styles[im_level])
            plt.scatter(x, y, color="k", s=16)

        plt.grid(True, which="major", ls="--", color="dimgrey")
        plt.xlim([1e-3, 10])
        plt.ylim([1e-6, 1])
        plt.xlabel(r"Intensity measure, $s$, [g]", fontsize=FONTSIZE)
        plt.ylabel("Annual probability of \n" + r"exceedance, $H(s)$", fontsize=FONTSIZE)
        plt.legend(frameon=False, loc='lower left', fontsize=FONTSIZE)
        plt.rc('xtick', labelsize=FONTSIZE)
        plt.rc('ytick', labelsize=FONTSIZE)

        # # Some text
        # plt.text(0.38, 0.0217, r"$Sa(2.0)$", fontsize=FONTSIZE)
        # plt.text(0.006, 0.0046, r"$PGA$", fontsize=FONTSIZE)
        # plt.text(2.2, 0.3, "(c)", fontsize=FONTSIZE)

        plt.show()

        return fig

    def predicted_vs_observed(self):
        # Plotting
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        styles = {}

        im_cnt = 0
        for path in self.fit_directory.iterdir():
            file = path.stem
            method = int(file[0])
            im_level = file[2:]

            if im_level != "PGA":
                continue

            name = get_model_label(method)

            if method not in styles:
                styles[method] = name
                label = styles[method]
            else:
                label = None

            if im_level not in styles:
                styles[im_level] = linestyles[im_cnt]
                im_cnt += 1

            data = json.load(open(path))
            x = data["x"]
            y = data["y"]
            x_fit = data["x_fit"]
            y_fit = data["y_fit"]

            interpolation = interp1d(x_fit, y_fit)
            y_fit_inter = interpolation(x)

            plt.plot(x, y / y_fit_inter, label=label, color=alt_color_grid[method - 1], ls=styles[im_level])

        plt.grid(True, which="major", ls="--", color="dimgrey")
        plt.xlim([1e-3, 10])
        plt.ylim([0, 2])
        plt.yticks([0, 0.5, 1.0, 1.5, 2.0])
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(r"Intensity measure, $s$, [g]", fontsize=FONTSIZE)
        plt.ylabel(r"$H(s)_{observed} / H(s)_{predicted}$", fontsize=FONTSIZE)
        # plt.legend(frameon=False, loc='lower left', fontsize=FONTSIZE)
        plt.rc('xtick', labelsize=FONTSIZE)
        plt.rc('ytick', labelsize=FONTSIZE)

        plt.text(0.0015, 1.8, "(c)", fontsize=FONTSIZE)

        plt.show()

        return fig

    def demand_intensity_model(self, model):
        m_low, b_low, m_up, b_up, limit = model
        if limit is not None:
            x_range = np.linspace(0, limit, 20)
        else:
            x_range = np.linspace(0, 4, 20)
        y_range = (x_range / m_low) ** (1 / b_low)

        # Plotting
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(x_range, y_range, color=color_grid[2])
        plt.text(0.3, 0.4, r"$\theta={%.2f}s^{%.2f}$" % (m_low, b_low), color=color_grid[2])

        # Upper branch if exists
        if limit is not None:
            x_range1 = np.linspace(limit, 4.0, 30)
            y_range1 = (x_range1 / m_up) ** (1 / b_up)
            plt.plot(x_range1, y_range1, color=color_grid[2])

            plt.plot([limit, limit], [0, 1], color="dimgrey", ls="--")
            plt.text(0.07, -0.12, r"$0.11%$", color="dimgrey")
            plt.text(0.35, 0.50, r"$\theta={%.2f}s^{%.2f}$" % (m_up, b_up), color=color_grid[2])

        plt.xlabel(r"Peak storey drift, $\theta$ [%]", fontsize=FONTSIZE)
        plt.ylabel("Intensity measure, " + r"$s$ [g]", fontsize=FONTSIZE)
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)

        plt.show()
        return fig

    def demand_exceedance_rates(self, model, beta):
        edp_range = np.linspace(0.01, 2., 50)

        # Plotting
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        styles = {}

        im_cnt = 0
        for path in self.fit_directory.iterdir():
            file = path.stem
            method = int(file[0])
            im_level = file[2:]

            if im_level != "PGA":
                continue

            name = get_model_label(method)

            if method not in styles:
                styles[method] = name
                label = styles[method]
            else:
                label = None

            styles[im_level] = linestyles[im_cnt]
            im_cnt += 1

            # read the model data
            data = json.load(open(path))
            x = data["x"]
            y = data["y"]
            x_fit = data["x_fit"]
            y_fit = data["y_fit"]
            coefs = data["coef"]

            # Compute MAFEs
            if method != 4:
                mafes_direct, mafes_model = \
                    compute_exceedance_rates(di_model, None, None, x, y, coefs, beta, edp_range)
                mafes_direct = np.array(mafes_direct)
                mafes_model = np.array(mafes_model)
            else:
                mafes_direct, mafes_model = \
                    compute_exceedance_rates(di_model, None, None, x_fit, y_fit, coefs, beta, edp_range)
                mafes_model = np.array(mafes_direct)

            # Direct method
            if im_cnt == 1:
                plt.plot(edp_range, mafes_direct, color='k', label="Direct Integration")

            # models
            plt.plot(edp_range, mafes_model, label=label, color=alt_color_grid[method-1], ls=styles[im_level])

        plt.yscale("log")
        plt.grid(True, which="major", ls="--", color="dimgrey")
        plt.xlim([1e-2, 2])
        plt.ylim([1e-5, 1])
        plt.xlabel(r"Peak storey drift, $\theta$ [%]", fontsize=FONTSIZE)
        plt.ylabel(r"MAF $\lambda$", fontsize=FONTSIZE)

        # plt.legend(frameon=False, loc='upper right')
        plt.text(0.075, 1.5e-5, "(c)", fontsize=FONTSIZE)

        plt.show()

        return fig


if __name__ == "__main__":
    path = Path.cwd()
    fitted_hazard_path = path / "outputs" / "usa"

    di_model = [0.5, 1.0, None, None, None]

    hazard = HazardPlots(fitted_hazard_path)

    # # Hazards
    # fig = hazard.plot_hazards()
    # plot_as_emf(fig, filename=path / "figs/hazards_high")

    # # Fitted vs observed
    # fig = hazard.predicted_vs_observed()
    # plot_as_emf(fig, filename=path / "figs/predicted_vs_observed_C")

    # # Demand intensity model
    # fig = hazard.demand_intensity_model(di_model)
    # plot_as_emf(fig, filename=path / "figs/demand_intensity_model")

    # Demand exceedance rates
    fig = hazard.demand_exceedance_rates(di_model, .4)
    plot_as_emf(fig, filename=path / "figs/mafes_usa")
