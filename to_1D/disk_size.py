import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from astropy.constants import G, M_sun
import astropy.units as u
from ..main import dataclass

def calc_disksize(self, 
                  use_fitted_H = False,
                  h = 20, 
                  r_in = 10, 
                  r_out = 500, 
                  n_bins = 200, 
                  a = 0.8, 
                  plot = True, 
                  avg_cells = 10, 
                  verbose = 1):
    
    try: self.cyl_z
    except: self.recalc_L()
    
    if use_fitted_H:
        try: self.r_bins
        except: 
            self.fit_HΣ(r_in = r_in, 
                        r_out = r_out, 
                        n_bins = n_bins, 
                        plot = False, 
                        verbose=verbose)
    if use_fitted_H:
        rad_bins = self.r_bins
        r_plot = rad_bins[:-1] + 0.5 * np.diff(rad_bins)
        H_func = interp1d(self.H_1D[:,0], r_plot, fill_value='extrapolate')

        mask_r = (self.cyl_R > rad_bins.min()) & (self.cyl_R < rad_bins.max())
        mask_h = abs(self.cyl_z[mask_r]) < 3 * H_func(self.cyl_R[mask_r])
        mask = np.zeros_like(mask_r, dtype = 'bool')
        mask[mask_r] = mask_h
    
    else:
        h, r_in, r_out = np.array([h, r_in, r_out]) / self.code2au
        rad_bins = np.logspace(np.log10(r_in), np.log10(r_out), n_bins)
        r_plot = rad_bins[:-1] + 0.5 * np.diff(rad_bins)    
        mask = (abs(self.cyl_z) <  h) & (self.cyl_R < r_out)
    
    vφ = np.sum(self.vrel[:,mask] * self.e_phi[:,mask], axis=0)
    m = self.m[mask]
    R = self.cyl_R[mask]

    h_mass, _ = np.histogram(R, bins = rad_bins, weights =  m)
    h_vφ, _ = np.histogram(R, bins = rad_bins, weights =  vφ * m)
    h_vφ2, _ = np.histogram(R, bins = rad_bins, weights =  vφ**2 * m)

    vφ_1D = (h_vφ/h_mass) 
    vφ2 = (h_vφ2/h_mass) 
    σvφ_1D = np.sqrt(vφ2 - vφ_1D**2) 
    self.vφ_1D = np.stack((vφ_1D, σvφ_1D), axis = 1)

    ####### Include self-gravity from the disk #######
    origo_bins = np.insert(rad_bins, 0, 0)[:-1]
    annulus_mass, _ = np.histogram(np.linalg.norm(self.rel_xyz, axis = 0), bins = origo_bins, weights=self.mhd['m'])
    accumulated_mass = np.cumsum(annulus_mass)

    self.kep_vel = (((G * ((self.sink_mass + accumulated_mass)  * self.m_cgs) * u.g) / (r_plot * self.code2au * u.au))**0.5).to('cm/s').value

    orbitvel_ratio_mean = uniform_filter1d(self.v_cgs * self.vφ_1D[:,0] / self.kep_vel, size = avg_cells)
    orbitvel_ratio_mean_sigma = uniform_filter1d(self.v_cgs * self.vφ_1D[:,1] / self.kep_vel, size = avg_cells)
    for i in range(len(self.vφ_1D[:,0])):
        if orbitvel_ratio_mean[i] < a:
            self.disk_size = r_plot[i] * self.code2au
            if verbose > 0: print(f'Disk size: {self.disk_size:2.1f} au')
            break
    try: 
        self.disk_size
    except: 
        self.disk_size = np.nan
        if verbose > 0: print('No disk size found')

    if plot:
        fig, axs = plt.subplots(1, 2, figsize = (20,6),gridspec_kw={'width_ratios': [2, 1.5]})


        axs[0].loglog(r_plot * self.code2au, self.kep_vel, label = 'Keplerian Orbital Velocity', color = 'black')
        axs[0].loglog(r_plot * self.code2au, self.vφ_1D[:,0]* self.v_cgs , label = 'Azimuthal velocity v$_φ$', c = 'blue')
        axs[0].fill_between(r_plot * self.code2au, (self.vφ_1D[:,0]- self.vφ_1D[:,1]) * self.v_cgs, (self.vφ_1D[:,0]+ self.vφ_1D[:,1])* self.v_cgs, alpha = 0.5, label = '$\pm1\sigma_{φ}$')

        axs[0].set(xlabel = 'Distance from sink [au]', ylabel = 'Orbital speed [cm/s]')

        axs[0].legend(frameon = False)
        axs[1].semilogx(r_plot * self.code2au, orbitvel_ratio_mean, label = 'v$_φ$/v$_K$ ratio', color = 'black', lw = 0.8)
        axs[1].fill_between(r_plot * self.code2au, orbitvel_ratio_mean - orbitvel_ratio_mean_sigma, orbitvel_ratio_mean + orbitvel_ratio_mean_sigma, alpha = 0.5, color = 'grey', label = '$\pm1\sigma_{v_φ/v_K}$')
        axs[1].axhline(a, color = 'red', ls = '--', label = f'a = {a}')
        axs[1].axhline(1, color = 'black', ls = '-', alpha = 0.7)
        axs[1].set(xlabel = 'Distance from sink [au]', ylim = (0.5, 1.1))
        axs[1].legend(frameon = False)
    
    
dataclass.calc_disksize = calc_disksize