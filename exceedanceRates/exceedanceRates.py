import numpy as np
from scipy.optimize import optimize
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from utils import mlefit


def get_probabilities_of_exceedance(edp, eta):

    counts = []
    for i in range(len(edp)):
        count = sum(val >= eta for val in edp[i])
        counts.append(count)
    return counts


class ExceedanceRates:
    def __init__(self, pars, msa, im, sa_hazard, apoe_hazard, coefs_hazard, eta, beta, beta_du_im=0.3):
        """

        Parameters
        ----------
        pars: List[float]
            m_low, b_low, m_up, b_up, limit
            Demand-intensity model fitting parameters (linear or bilinear)
        msa: dict
            MSA outputs
        im: List
            IM range
        sa_hazard: List
            Spectral accelerations of seismic hazard
        apoe_hazard: List
            Annual probability of exceedances of seismic hazard
        coefs_hazard: List
            Coefficients of SAC/FEMA-compatible hazard model
        eta: float
            EDP level for which exceedance rates are being computed for
        beta: float
            Dispersion of EDP capacity
        beta_du_im: float
            Inherent (epistemic) uncertainty of IM
        """
        self.pars = pars
        self.msa = msa
        self.im = im
        self.eta = eta
        self.sa_hazard = sa_hazard
        self.apoe_hazard = apoe_hazard
        self.coefs_hazard = coefs_hazard
        self.beta_edp = beta
        self.beta_du_im = beta_du_im

    def direct_integration(self, sa_hazard, hazard, beta_dr_im=None, limit=None):
        """

        Parameters
        ----------
        sa_hazard: List
            Spectral accelerations of seismic hazard
        hazard: List
            Annual probability of exceedances of seismic hazard
        beta_dr_im: float
            Natural (aleatory) randomness of IM
        limit: float
            EDP or IM limit to separate the two lines of the bilinear model

        Returns
        -------
        l: float
            MAFE from direct integration
        beta_dr_im: float
        """
        m_low, b_low, m_up, b_up, _ = self.pars

        if self.im is not None:
            theta_hat_mom = np.exp(np.mean(np.log(self.im)))
            beta_hat_mom = np.std(np.log(self.im))
            x0 = [theta_hat_mom, beta_hat_mom]
            counts = get_probabilities_of_exceedance(self.msa, self.eta)

            im = np.linspace(min(self.im[:]), max(self.im[:]), 200)
            interpolation = interp1d(self.im[:], counts[:])
            new_counts = interpolation(im)

            # Fit a lognormal distribution
            fit_counts = np.array(counts[:])
            fit_ims = self.im[:]

            xopt_mle = optimize.fmin(func=lambda var: mlefit(theta=[var[0], var[1]], num_recs=len(self.msa[0]),
                                                             num_collapse=fit_counts, IM=fit_ims),
                                     x0=x0, maxiter=3000, maxfun=3000, disp=False)
            eta_d = xopt_mle[0]

            if beta_dr_im is None:
                beta_dr_im = xopt_mle[1]
        else:

            if limit is not None:
                edplimit = m_low * limit ** b_low

                if self.eta < edplimit:
                    eta_d = (self.eta / m_low) ** (1 / b_low)
                else:
                    eta_d = (self.eta / m_up) ** (1 / b_up)
            else:
                eta_d = (self.eta / m_low) ** (1 / b_low)

        # Do first strip
        s_f = []
        H_f = []
        for aa, bb in zip(sa_hazard, hazard):
            if bb > 0:
                s_f.append(aa)
                H_f.append(bb)

        # Do second strip
        s_ff = []
        H_ff = []
        for i in range(len(s_f) - 1):
            if H_f[i] - hazard[i + 1] > 0:
                s_ff.append(s_f[i])
                H_ff.append(H_f[i])
        s_ff.append(s_f[-1])
        H_ff.append(H_f[-1])

        # Overwrite the initial variable for convenience
        s = s_ff
        H = H_ff

        if limit is not None and eta_d >= limit:
            b = b_up
        else:
            b = b_low

        if beta_dr_im is None:
            beta_dr_im = 0

        beta = np.sqrt((self.beta_du_im / b) ** 2 + (self.beta_edp / b) ** 2 + beta_dr_im ** 2)

        # First we compute the PDF value of the fragility at each of the discrete
        # hazard curve points
        p = lognorm.cdf(s, beta, scale=eta_d)

        # This function computes the MAF using Method 1 outlined in
        # Porter et al. [2004]
        # This assumes that the hazard curve is linear in logspace between
        # discrete points among others

        # Initialise some arrays
        ds = []
        ms = []
        dHds = []
        dp = []
        dl = []

        for i in np.arange(len(s) - 1):
            ds.append(s[i + 1] - s[i])
            ms.append(s[i] + ds[i] * 0.5)
            dHds.append(np.log(H[i + 1] / H[i]) / ds[i])
            dp.append(p[i + 1] - p[i])
            dl.append(p[i] * H[i] * (1 - np.exp(dHds[i] * ds[i])) - dp[i] / ds[i] * H[i] * (
                    np.exp(dHds[i] * ds[i]) * (ds[i] - 1 / dHds[i]) + 1 / dHds[i]))

        # Compute the MAFE
        l = sum(dl)

        return l, beta_dr_im

    def simplified_model_bilinear(self, edptype="drift", beta_dr_im=None):
        """

        Parameters
        ----------
        edptype: str
            Engineering demand parameter type
        beta_dr_im: float
            Natural (aleatory) randomness of IM

        Returns
        -------
        mafe: float
            MAFE using the seimsic hazard model
        l: float
            MAFE from direct integration
        beta_dr_im: float
        """
        m_low, b_low, m_up, b_up, edplimit = self.pars

        if edptype == "drift":
            limit = (edplimit / m_low) ** (1 / b_low)
        else:
            limit = edplimit

        l, beta_dr_im = self.direct_integration(self.sa_hazard, self.apoe_hazard,
                                                beta_dr_im=beta_dr_im, limit=limit)

        # Transform dispersions to EDP based quantities assuming homoscedasticity
        beta_rdr_lower = beta_dr_im * b_low
        beta_d_lower = self.beta_du_im
        beta_rdr_upper = beta_dr_im * b_up
        beta_d_upper = self.beta_du_im

        im_lower = (self.eta / m_low) ** (1 / b_low)
        im_upper = (self.eta / m_up) ** (1 / b_up)

        # Computed later as beta_rdr (from msa)
        beta_lower_total = beta_d_lower ** 2 + self.beta_edp ** 2 + beta_rdr_lower ** 2
        beta_upper_total = beta_d_upper ** 2 + self.beta_edp ** 2 + beta_rdr_upper ** 2

        k0, k1, k2 = self.coefs_hazard

        H_lower = k0 * np.exp(-k1 * np.log(im_lower) - k2 * np.log(im_lower) ** 2)
        H_upper = k0 * np.exp(-k1 * np.log(im_upper) - k2 * np.log(im_upper) ** 2)

        phi_lower = 1 / (1 + 2 * k2 / b_low ** 2 * beta_lower_total)
        phi_upper = 1 / (1 + 2 * k2 / b_up ** 2 * beta_upper_total)

        sigma_lower = beta_lower_total * np.sqrt(phi_lower) / b_low
        sigma_upper = beta_upper_total * np.sqrt(phi_upper) / b_up

        mu_lower = phi_lower * ((np.log(self.eta) - np.log(m_low)) / b_low - k1 * beta_lower_total / b_low ** 2)
        mu_upper = phi_upper * ((np.log(self.eta) - np.log(m_up)) / b_up - k1 * beta_upper_total / b_up ** 2)

        F_lower = lognorm(s=sigma_lower, scale=np.exp(mu_lower)).cdf(limit)
        F_upper = lognorm(s=sigma_upper, scale=np.exp(mu_upper)).cdf(limit)

        exp_term_lower = np.exp(k1 ** 2 * phi_lower / (2 * b_low ** 2) * beta_lower_total)
        exp_term_upper = np.exp(k1 ** 2 * phi_upper / (2 * b_up ** 2) * beta_upper_total)

        G_lower = np.sqrt(phi_lower) * k0 ** (1 - phi_lower) * H_lower ** phi_lower * exp_term_lower
        G_upper = np.sqrt(phi_upper) * k0 ** (1 - phi_upper) * H_upper ** phi_upper * exp_term_upper
        mafe = F_lower * G_lower + (1 - F_upper) * G_upper

        return mafe, l, beta_dr_im

    def simplified_model_linear(self, beta_dr_im=None, run_direct=True):
        """

        Parameters
        ----------
        beta_dr_im: float
            Natural (aleatory) randomness of IM
        run_direct: bool
            Run direct integration?

        Returns
        -------
        mafe: float
            MAFE using the seimsic hazard model
        l: float
            MAFE from direct integration
        beta_dr_im: float

        """
        l = None
        if beta_dr_im is None:
            beta_dr_im = 0.
        m_low, b_low, _, _, _ = self.pars

        if run_direct:
            l, beta_dr_im = self.direct_integration(self.sa_hazard, self.apoe_hazard,
                                                    beta_dr_im=beta_dr_im)

        # Transform dispersions to EDP based quantities assuming homoscedasticity
        beta_rdr = beta_dr_im * b_low
        beta_d = self.beta_du_im

        im_level = (self.eta / m_low) ** (1 / b_low)

        beta = beta_d ** 2 + self.beta_edp ** 2 + beta_rdr ** 2

        if len(self.coefs_hazard) == 3:
            k0, k1, k2 = self.coefs_hazard

            H = k0 * np.exp(-k1 * np.log(im_level) - k2 * np.log(im_level) ** 2)

            phi_prime = 1 / (1 + 2 * k2 / (b_low ** 2) * beta)
            exp_term = np.exp(k1 ** 2 * phi_prime / (2 * b_low ** 2) * beta)
            mafe = np.sqrt(phi_prime) * k0 ** (1 - phi_prime) * H ** phi_prime * exp_term

        else:
            k0, k = self.coefs_hazard

            H = k0 * np.exp(-k * np.log(im_level))

            exp_term = np.exp(k ** 2 / (2 * b_low ** 2) * beta)
            mafe = H * exp_term

        return mafe, l, beta_dr_im
