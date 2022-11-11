import os
from EzGM.utility import hazard_curve, disagg_MR, disagg_MReps, get_available_gmpes, check_gmpe_attributes, \
    parse_sa_lt_to_avgsa

from utils import create_dir


class PSHA:
    mag_bin_width = None
    distance_bin_width = None
    reference_vs30_value = None
    poes = None
    results_dir = None

    def __init__(self, oq_model, oq_ini, post_dir):
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
        """
        self.oq_model = oq_model
        self.oq_ini = oq_ini
        self.post_dir = post_dir

    def read_ini_file(self):
        """
        Reads job.ini file
        Returns
        -------
        None
        """
        with open(os.path.join(self.oq_model, self.oq_ini)) as f:
            info = f.readlines()
            for line in info:
                if line.startswith('poes'):
                    self.poes = [float(poe) for poe in
                                line.split('\n')[0].split('=')[1].split(',')]
                if line.startswith('export_dir'):
                    self.results_dir = os.path.join(oq_model, line.split('\n')[0].split('=')[1].strip())
                if line.startswith('mag_bin_width'):
                    self.mag_bin_width = float(line.split('=', 1)[1])
                if line.startswith('distance_bin_width'):
                    self.distance_bin_width = float(line.split('=', 1)[1])
                if line.startswith('reference_vs30_value'):
                    self.reference_vs30_value = float(line.split('=', 1)[1])

    def run_psha(self):
        """
        Runs PSHA
        Returns
        -------
        None
        """
        # Create the export directory for analysis results
        create_dir(self.results_dir)

        cwd = os.getcwd()  # Current working directory
        os.chdir(oq_model)  # Change directory, head to OQ_model folder
        os.system('oq engine --run ' + oq_ini + ' --exports csv')
        os.chdir(cwd)  # go back to the previous working directory

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

    @staticmethod
    def parse_lt_file_to_avgsa(xml_file, out_file, periods, corr_method):
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


if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.realpath(""))
    oq_model = os.path.join(parent_path, 'Hazard', 'data')
    oq_ini = "AvgSa_3.ini"
    post_dir = "OQproc_Outputs"

    psha = PSHA(oq_model, oq_ini, post_dir)
    # psha.derive_openquake_info()
    psha.read_ini_file()
    psha.run_psha()
