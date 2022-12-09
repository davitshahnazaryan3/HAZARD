"""
TODO: fix EZGM
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from EzGM.utility import hazard_curve, disagg_MR, disagg_MReps, get_available_gmpes, check_gmpe_attributes
from Hazard.psha import PSHA
from EzGM.selection import conditional_spectrum
from utils import create_folder


def run_ground_motion_selection(oq_model_path, oq_ini_filename, export_path, ref_periods,
                                run_psha=True, record_selection=True, plot_flag=False,
                                Tstar=None, ngm=25, database_dir=None, get_disaggregation_results=False,
                                ims=None):

    post_full_path = oq_model_path / export_path
    create_folder(post_full_path)

    if run_psha:
        # Hazard Analysis via OpenQuake
        psha = PSHA(oq_model_path, oq_ini_filename, export_path, ref_periods)

        # psha.derive_openquake_info('BooreAtkinson2008')
        psha.read_ini_file()
        periods = psha.get_period_ranges_for_sa_avg()
        psha.parse_lt_file_to_avgsa(
            oq_model_path / "gmmLT.xml",
            oq_model_path / "gmmLT_sa_avg.xml",
            periods,
        )

        psha.run_psha()

        # Extract and plot hazard curves in a reasonable format
        if plot_flag:
            psha.get_figures()

    if record_selection:
        if Tstar is None:
            Tstar = np.arange(0.1, 1.1, 0.1)

        if not run_psha:
            psha = PSHA(oq_model_path, oq_ini_filename, export_path, ref_periods)
            psha.read_ini_file()

            if get_disaggregation_results:
                hazard_curve(psha.poes, psha.results_dir.as_posix(), post_full_path.as_posix(), show=0)
                disagg_MR(psha.mag_bin_width, psha.distance_bin_width, psha.results_dir.as_posix(), post_full_path,
                          n_rows=3, show=0)
                disagg_MReps(psha.mag_bin_width, psha.distance_bin_width, psha.results_dir.as_posix(), post_full_path,
                             n_rows=3, show=0)

        # Record Selection
        if ims is None:
            ims = []
            for filename in post_full_path.iterdir():
                file = filename.stem
                if file.startswith('imls'):
                    ims.append(file.split('_')[1].split('.out')[0])

        for im in ims:  # for each im in the im list
            # read hazard and disaggregation info
            imls = np.loadtxt(oq_model_path / export_path / f'imls_{im}.out')
            mean_mags = np.loadtxt(oq_model_path / export_path / f'mean_mags_{im}.out')
            mean_dists = np.loadtxt(oq_model_path / export_path / f'mean_dists_{im}.out')

            for i in range(len(psha.poes)):
                # 1. Initialize the conditional_spectrum object for record selection, check which parameters are
                # required for the gmpe you are using.
                cs = conditional_spectrum(database='NGA_W2',
                                          outdir=oq_model_path / f'EzGM_Outputs_{im}_POE-{psha.poes[i]}-in-50-years')

                # 2. Create target spectrum
                cs.create(Tstar=Tstar, gmpe='BooreEtAl2014', selection=1, Sa_def='RotD50',
                          site_param={'vs30': psha.reference_vs30_value},
                          rup_param={'rake': [0.0], 'mag': [mean_mags[i]]},
                          dist_param={'rjb': [mean_dists[i]]}, Hcont=None, T_Tgt_range=[0.05, 4.0],
                          im_Tstar=imls[i], epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram')

                # 3. Select the ground motions
                cs.select(nGM=ngm, isScaled=1, maxScale=2.5,
                          Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, nTrials=20,
                          weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=3, tol=10)

                if plot_flag:
                    # Plot the target spectrum, simulated spectra and spectra of selected records
                    cs.plot(tgt=0, sim=0, rec=1, save=1, show=0)
                    plt.close('all')

                # 4. If database == 'NGA_W2' you can first download the records via nga_download method from NGA-West2
                # Database [http://ngawest2.berkeley.edu/] and then use write method cs.ngaw2_download(username =
                # 'example_username@email.com', pwd = 'example_password123456', sleeptime = 2, browser = 'chrome')

                # 5. If you have records already inside recs_f\database.zip\database or
                # downloaded records for database = NGA_W2 case, write whatever you want,
                # the object itself, selected and scaled time histories
                cs.write(obj=1, recs=0, recs_f=database_dir.as_posix())


if __name__ == "__main__":
    # Set path to OpenQuake model .ini file path
    parent_path = Path.cwd().parents[2] / "Projects/Data-Driven Design"

    oq_model = parent_path / "PSHA/SaT"
    oq_ini = "SaT.ini"
    export_path = "OQproc_Outputs"
    ref_periods = [1.0]
    database_dir = parent_path.parents[0]

    run_ground_motion_selection(oq_model, oq_ini, export_path, ref_periods,
                                run_psha=False, record_selection=True, Tstar=1, ngm=20,
                                database_dir=database_dir, ims=['SA(1.0)'])
