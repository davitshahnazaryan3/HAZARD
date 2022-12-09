"""
Performs Probabilistic seismic hazard assessment
"""

from pathlib import Path
from EzGM.utility import hazard_curve, disagg_MR, disagg_MReps, get_available_gmpes, check_gmpe_attributes, \
    parse_sa_lt_to_avgsa
import subprocess
from utils import create_dir, get_range
import logging

logging.basicConfig(filename="../.logs/logs_psha.txt",
                    level=logging.DEBUG,
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")


def derive_openquake_info(gmpe=None):
    """
    Gets some info on GMPEs available in the OpenQuake engine
    Parameters
    ----------
    gmpe: str
        Name of the GMPE for which attributes are sought for

    Returns
    -------
    gmpes: List
        List of GMPEs available in the OpenQuake engine

    """
    gmpes = get_available_gmpes()

    if gmpe is not None:
        check_gmpe_attributes(gmpe)

    return gmpes


class PSHA:
    mag_bin_width = None
    distance_bin_width = None
    reference_vs30_value = None
    poes = None
    results_dir = None

    def __init__(self, oq_model, oq_ini, post_dir, ref_periods):
        """
        Probabilistic seismic hazard assessment
        Parameters
        ----------
        oq_model:
            Path of folder where OpenQuake input data is located
        oq_ini: str
            job.ini filename for OpenQuake to use
        post_dir:
            Path to export figures to
        ref_periods: List
            Reference periods [s], e.g. 0.5, [0.5, 1.0]
        """
        self.oq_model = oq_model
        self.oq_ini = oq_ini
        self.post_dir = post_dir
        self.ref_periods = ref_periods

    def read_ini_file(self):
        """
        Reads job.ini file
        Returns
        -------
        None
        """
        logging.info("Reading job file")

        try:
            with open(self.oq_model / self.oq_ini) as f:
                try:
                    info = f.readlines()
                    for line in info:
                        if line.startswith('poes'):
                            self.poes = [float(poe) for poe in
                                        line.split('\n')[0].split('=')[1].split(',')]
                        if line.startswith('export_dir'):
                            self.results_dir = self.oq_model / line.split('\n')[0].split('=')[1].strip()
                        if line.startswith('mag_bin_width'):
                            self.mag_bin_width = float(line.split('=', 1)[1])
                        if line.startswith('distance_bin_width'):
                            self.distance_bin_width = float(line.split('=', 1)[1])
                        if line.startswith('reference_vs30_value'):
                            self.reference_vs30_value = float(line.split('=', 1)[1])
                except (IOError, OSError):
                    logging.error("Error while reading file")

        except (FileNotFoundError, PermissionError, OSError):
            logging.error("Error opening file")

    def run_psha(self):
        """
        Runs PSHA
        Returns
        -------
        None
        """
        # Create the export directory for analysis results
        create_dir(self.results_dir)

        logging.info("Running OpenQuake Engine")
        subprocess.call(['oq', 'engine', '--run', self.oq_model / self.oq_ini, '--exports', 'csv'])

        try:
            subprocess.check_output("ls non_existent_file; exit 0",
                                    stderr=subprocess.STDOUT,
                                    shell=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"Error running OpenQuake Engine: {e}")
            return

        logging.info("Completed Probabilistic Seismic Hazard Analysis")

    def get_figures(self):
        """
        Plots and exports figures of PSHA results
        Returns
        -------
        None
        """
        # Create the directory for processed results
        create_dir(self.post_dir)

        # Extract and plot hazard curves in a reasonable format
        hazard_curve(self.poes, self.results_dir, self.post_dir)

        # Extract and plot disaggregation results by M and R
        disagg_MR(self.mag_bin_width, self.distance_bin_width, self.results_dir, self.post_dir, n_rows=3)

        # Extract and plot disaggregation results by M, R and epsilon
        disagg_MReps(self.mag_bin_width, self.distance_bin_width, self.results_dir, self.post_dir, n_rows=3)

    @staticmethod
    def parse_lt_file_to_avgsa(xml_file, out_file, periods, corr_method='baker_jayaram'):
        """

        Parameters
        ----------
        xml_file : str
            Input GMPE logic tree for SA, e.g. 'gmmLT.xml'
        out_file : str
            The output GMPE LT file, e.g. 'gmmLT_AvgSA.xml'
        periods : List[List]
            List of periods for the AvgSA calculation
            e.g. periods = [[0.4,0.5,0.6,0.7,0.8], [1.1,1.2,1.3,1.4,1.5]]
        corr_method: str
        String for one of the supported correlation models (e.g. 'akkar', 'baker_jayaram')
        Returns
        -------
        None
        """
        parse_sa_lt_to_avgsa(xml_file, out_file, periods, corr_method)

    def get_period_ranges_for_sa_avg(self):
        periods = []
        for period in self.ref_periods:
            periods.append(get_range(period))
        return periods


if __name__ == "__main__":
    parent_path = Path("E:/") / "Data-Driven Design/PSHA"
    #
    # oq_model = parent_path / "AvgSa"
    # oq_ini = "AvgSa.ini"
    # post_dir = "OQproc_Outputs"
    # # ref_periods = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    # ref_periods = [3.0]
    #
    # psha = PSHA(oq_model, oq_ini, post_dir, ref_periods)
    #
    # # psha.derive_openquake_info('BooreAtkinson2008')
    # psha.read_ini_file()
    # periods = psha.get_period_ranges_for_sa_avg()
    # psha.parse_lt_file_to_avgsa(
    #     oq_model / "gmmLT.xml",
    #     oq_model / "gmmLT_sa_avg.xml",
    #     periods,
    # )
    # psha.run_psha()
