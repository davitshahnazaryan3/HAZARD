"""
Software to obtain response spectra from a record.
"""
from __future__ import division

from typing import List

import pandas as pd
import numpy as np
from pathlib import Path


class ResponsSpectrumFromGM:
	T = np.arange(0, 4.01, 0.01)
	RS = {'T1': T}

	def __init__(self, damping, export=False):
		self.damping = damping
		self.export = export

	def get_sa(self, acc, dt, period, damping):
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
		H[np.int_(np.arange(1, symIdx))] = np.array([natFreq ** 2 * 1 / ((natFreq ** 2 - i ** 2) + 2 * 1j * damping *
		                                                                 i * natFreq) for i in freq[1:symIdx]])

		if nPts % 2 != 0:
			H[np.int_(np.arange(len(H) - symIdx + 1, len(H)))] = np.flipud(np.conj(H[np.int_(np.arange(1, symIdx))]))
		else:
			H[np.int_(np.arange(len(H) - symIdx + 2, len(H)))] = np.flipud(np.conj(H[np.int_(np.arange(1, symIdx - 1))]))

		sa = max(abs(np.real(np.fft.ifft(np.multiply(H, fas)))))
		return sa

	def derive_response_spectrum(self, dt_filepath, gm_filepath):

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
			Sa = np.zeros(len(self.T))
			for j in range(len(self.T)):
				Sa[j] = self.get_sa(acc, dt, self.T[j], self.damping)
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

	rs = ResponsSpectrumFromGM(damping)
	rs.derive_response_spectrum(dt_filepath, gm_filepath)
