"""
Software to obtain response spectra from a record.
"""
from __future__ import division

import functools
from typing import List

import pandas as pd
import numpy as np
from pathlib import Path


class ResponseSpectrumFromGM:
    # Periods
    T = np.arange(0, 4.01, 0.01)
    # Response spectra
    RS = {'T1': T}

    def __init__(self, damping, export=False):
        """

        Parameters
        ----------
        damping: float
            Damping ratio
        export: bool
            Export response spectrum to .pickle and .csv?
        """
        self.damping = damping
        self.export = export

    def get_sa(self, period, acc, dt):
        """
        Get spectral acceleration at a period
        Parameters
        ----------
        period: float
            Period at which to calculate the SA
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram

        Returns
        -------
        sa: float
            Spectral acceleration at the given period

        """
        if period == 0.0:
            # peak ground acceleration, PGA
            period = 1e-20

        pow = 1
        while 2 ** pow < len(acc):
            pow = pow + 1

        nPts = 2 ** pow
        fas = np.fft.fft(acc, nPts)
        dFreq = 1 / (dt * (nPts - 1))
        freq = dFreq * np.array(range(nPts))
        if nPts % 2 != 0:
            symIdx = int(np.ceil(nPts / 2))
        else:
            symIdx = int(1 + nPts / 2)

        natFreq = 1 / period
        H = np.ones(len(fas), 'complex')
        H[np.int_(np.arange(1, symIdx))] = np.array([natFreq ** 2 * 1 /
                                                     ((natFreq ** 2 - i ** 2) + 2 * 1j * self.damping * i * natFreq)
                                                     for i in freq[1:symIdx]])

        if nPts % 2 != 0:
            H[np.int_(np.arange(len(H) - symIdx + 1, len(H)))] = np.flipud(np.conj(H[np.int_(np.arange(1, symIdx))]))
        else:
            H[np.int_(np.arange(len(H) - symIdx + 2, len(H)))] = np.flipud(np.conj(H[np.int_(np.arange(1, symIdx - 1))]))

        sa = max(abs(np.real(np.fft.ifft(np.multiply(H, fas)))))
        return sa

    def derive_response_spectrum(self, dt_filepath, gm_filepath):
        """
        Derives response spectrum for 1 or more ground motion records
        Parameters
        ----------
        dt_filepath: Path
            Path to a file containing time steps of each ground motion of interest
        gm_filepath: Path or List[Path]
            Path to a file containing filenames of each ground motion of interest
        Returns
        -------
        None
        """

        if isinstance(gm_filepath, List):
            gm_files = []

            for file in gm_filepath:
                gm_files += list(pd.read_csv(file, header=None)[0])

        else:
            gm_files = list(pd.read_csv(gm_filepath, header=None)[0])

        dts = np.array(pd.read_csv(dt_filepath, header=None)[0])

        for i in range(len(dts)):
            print(gm_files[i])
            acc = np.array(pd.read_csv(gm_path / gm_files[i], header=None)[0]) * 4
            dt = dts[i]

            Sa = list(map(functools.partial(self.get_sa, acc=acc, dt=dt), self.T))

            self.RS[gm_files[i].replace('.txt', '')] = Sa

        self.RS = pd.DataFrame.from_dict(self.RS)

        if self.export:
            # Storing the RS dataframe into a csv file
            pd.DataFrame.to_csv(self.RS, 'RS.csv')
            self.RS.to_pickle('RS.pickle')


if __name__ == "__main__":
    damping = 0.05
    gm_path = Path.cwd() / 'share_GM'
    dt_filepath = gm_path / 'FEMA_P695_unscaled_dts.txt'
    gm_filepath = [gm_path / 'FEMA_P695_unscaled_names.txt']

    rs = ResponseSpectrumFromGM(damping)
    rs.derive_response_spectrum(dt_filepath, gm_filepath)
