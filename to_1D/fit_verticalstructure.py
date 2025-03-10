import numpy as np
import matplotlib.pyplot as plt
import tqdm, sys
from ..path_config import config
sys.path.insert(0, config['user_lmfit_path'])
from lmfit import Model  # type: ignore
from ..main import dataclass

def fit_HΣ(self, 
           r_in = 5, 
           r_out = 500, 
           n_bins = 500,
           h_bins_pct = 0.2, 
           plot = True, 
           MMSN = True, 
           dpi = 100, 
           verbose = 1, 
           validate_fit = False):
    
    try: self.cyl_z
    except: self.recalc_L()
    r_in /= self.code2au; r_out /= self.code2au

    r_bins = np.logspace(np.log10(r_in), np.log10(r_out), n_bins)
    r_1D = r_bins[:-1] + 0.5 * np.diff(r_bins) 

    def H_func(x, Σ, H): return (Σ) / (np.sqrt(2 * np.pi) * H) * np.exp( - x**2 / (2 * H**2)) 

    def fit_scaleheight(ρ, h, σ_ρ, x0):
        model = Model(H_func)
        params = model.make_params(Σ = x0[0], H = x0[1])
        result = model.fit(ρ, x = h, params = params, weights = σ_ρ, nan_policy='omit')
        fit_params = np.array(list(result.best_values.values()))
        fit_err = np.array([par.stderr for _, par in result.params.items()])
        return np.array([fit_params[0], fit_err[0]]), np.array([fit_params[1], fit_err[1]]) 

    mask = (self.cyl_R > r_in) & (self.cyl_R < r_out) & (abs(self.cyl_z) < 2 * r_out) 

    densities = {key: [] for key in range(n_bins - 1)}
    heights = {key: [] for key in range(n_bins - 1)}

    R_binID = np.digitize(self.cyl_R[mask], bins = r_bins) 

    #### The 0th bin only contains values at < r_in and is removed by the mask ####
    for bin in np.unique(R_binID[1:]):
        densities[bin - 1].extend(self.mhd['d'][mask][R_binID == bin])
        heights[bin - 1].extend(self.cyl_z[mask].flatten()[R_binID == bin])

    for key in densities: densities[key] = np.array(densities[key]); heights[key] = np.array(heights[key])

    self.Σ_1D = np.zeros((n_bins - 1, 2))
    self.H_1D = np.zeros((n_bins - 1, 2))
    self.r_bins = r_bins
    x0 = np.array([1e3 / self.Σ_cgs, 7 / self.code2au]) # Initial guess for surface density and scaleheight in code-units

    if verbose > 0: print('Fitting surface density and scaleheight in each radial bin')
    n_hbins = np.rint(n_bins * h_bins_pct).astype(int)
    h_bins = np.linspace(-r_out, r_out, n_hbins)
    h_plot = h_bins[:-1] + 0.5 * np.diff(h_bins) 
    #### The function is changed to also bin height data calc. mean and uncertainty. ####
    
    for i in tqdm.tqdm(range(n_bins - 1), disable = not self.loading_bar): 
        rho_mean = np.full_like(h_plot, np.nan)
        rho_sigma = np.full_like(h_plot, np.nan)
        H_binID = np.digitize(heights[i], bins = h_bins) 
        for j, H in enumerate(np.unique(H_binID)[1:-1]):
            rho_mean[H - 1] = densities[i][H == H_binID].mean();
            rho_sigma[H - 1] = densities[i][H == H_binID].std() / len(densities[i][H == H_binID])**0.5
        
        self.Σ_1D[i], self.H_1D[i] = fit_scaleheight(rho_mean, h_plot, rho_sigma, x0=x0)

    def check_HΣfit(nH):
        annulus_m_sum = np.zeros(n_bins - 1)
        annulus_V_sum = np.zeros(n_bins - 1)   

        for bin in np.unique(R_binID[1:]):
            h_bool = abs(self.cyl_z[mask][R_binID == bin]) < nH * (self.H_1D[bin - 1, 0])

            annulus_m_sum[bin - 1] = np.sum(self.m[mask][R_binID == bin][h_bool])
            annulus_V_sum[bin - 1] = np.sum(((self.amr['ds'])**3)[mask][R_binID == bin][h_bool]) 

        #np.pi*((radius + Δ_radius)**2 - (radius -  Δ_radius)**2) * H_p
        annulus_vol = np.pi * (np.roll(r_bins, -1)[:-1]**2 - r_bins[:-1]**2) * 2 * nH * self.H_1D[:, 0] 
        
        #Area of each annulus in [cm**2]              
        annulus_area = np.pi * (np.roll(r_bins, -1)[:-1]**2 - r_bins[:-1]**2)

        #Average of density from total cell mass over total cell volume                        
        annulus_mtot = annulus_m_sum / annulus_V_sum  * annulus_vol                                                

        Σ_calc = annulus_mtot / annulus_area
        
        return Σ_calc 
    
    if plot:
        if validate_fit:
            print('Validating fit...')
            sigmas = np.asarray([check_HΣfit(σ) for σ in range(1, 3)])
        fig, axs = plt.subplots(1,3, figsize = (20, 6), dpi = dpi)
        ax = axs[0]

        ax.loglog(r_1D * self.code2au, self.Σ_1D[:,0]* self.Σ_cgs, color = 'blue', label = 'Σ$_{Fit}$')
        for i in reversed(range(1, 3)):
            if validate_fit: ax.loglog(r_1D * self.code2au, sigmas[i - 1] * self.Σ_cgs, color = 'red', label = 'Σ$_{Calc}$'+f'$\propto\int\pm{i}H$', alpha = i/2, lw = 0.8)
        ax.fill_between(r_1D * self.code2au, (self.Σ_1D[:,0] + self.Σ_1D[:,1])* self.Σ_cgs, (self.Σ_1D[:,0] - self.Σ_1D[:,1])* self.Σ_cgs, alpha = 0.45, color = 'blue')
        ax.set(ylabel = 'Σ$_{gas}$ [g/cm$^2$]', xlabel = 'Distance from sink [au]', title = 'Surface density Σ$_{gas}$(r)')

        if MMSN:
            Σ_MMSN = lambda r: 1700 * (r)**(-3/2)
            r = r_1D * self.code2au
            #ax.text(r[0], Σ_MMSN(r)[0] - 25, 'Σ$_{MMSN}\propto r^{-3/2}$', va = 'top', ha = 'left', rotation = -26, color = 'grey')
            ax.loglog(r, Σ_MMSN(r), color = 'grey', ls = '--', label = 'Σ$_{MMSN}\propto r^{-3/2}$')

        ax.legend(frameon = False)

        ax = axs[1]
        ax.loglog(r_1D * self.code2au, self.H_1D[:,0] * self.code2au, label = 'Scale height H', color = 'green')
        ax.fill_between(r_1D * self.code2au, (self.H_1D[:,0] + self.H_1D[:,1]) * self.code2au, (self.H_1D[:,0] - self.H_1D[:,1]) * self.code2au, alpha = 0.3, color = 'green', label = '$\pm σ_H$')
        ax.set(ylabel = 'Scale height [au]', xlabel = 'Distance from sink [au]', title = 'Scale height  H(r)')
        ax.legend(frameon = False)

        ax = axs[2]
        open_angle =  np.vstack((np.arctan(self.H_1D[:,0] / r_1D) , np.arctan(self.H_1D[:,1] / r_1D) )).T

        ax.semilogx(r_1D * self.code2au, open_angle[:,0], color = 'purple', label = 'Opening angle H/r')
        ax.fill_between(r_1D * self.code2au, open_angle[:,0] + open_angle[:,1], open_angle[:,0] - open_angle[:,1], color = 'purple', alpha = 0.3, label = '$\pm σ_φ$')

        #Values for ticks
        values = np.linspace(0, np.pi/2, 5)
        names = ['$0$', '$π/8$', '$π/4$', '$3π/8$', '$π/2$']
        ax.set_yticks(values); ax.set_yticklabels(names)
        ax2 = ax.twinx()
        ax2.set_yticks(np.rad2deg(values))
        ax2.set_yticklabels([f'{deg:2.0f}'+'$^{\circ}$' for deg in np.rad2deg(values)])
        ax.set(ylabel = 'Opening angle [rad/deg]', xlabel = 'Distance from sink [au]', title = 'Opening angle H/r(r)')
        ax.legend(frameon = False)
        plt.tight_layout()


            


dataclass.fit_HΣ = fit_HΣ