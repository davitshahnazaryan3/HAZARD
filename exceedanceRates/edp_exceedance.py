import numpy as np

from exceedanceRates.exceedanceRates import ExceedanceRates


def compute_exceedance_rates(pars, response, im, s, apoe, coefs, beta_psd, edp_range=None,
                             beta_du_im=0.):
    if edp_range is None:
        edp_range = np.linspace(0.01, 2., 50)

    mafes_direct = []
    mafes_model = []

    for psd in edp_range:
        er = ExceedanceRates(pars, response, im, s, apoe, coefs, psd, beta_psd, beta_du_im)
        mafe, l, beta_dr_im = er.simplified_model_linear()
        mafes_model.append(mafe)
        mafes_direct.append(l)

    return mafes_direct, mafes_model
