import numpy as np


def get_probability_of_exceedance(return_period, investigation_time=50):
    poe = 1 - np.exp(-investigation_time / return_period)
    return poe


def get_annual_probability_of_exceedance(poe, investigation_time=50):
    apoe = -np.log(1 - poe) / investigation_time
    return apoe


def get_return_period(poe, investigation_time=50):
    return_period = 1 / get_annual_probability_of_exceedance(poe, investigation_time)
    return return_period

