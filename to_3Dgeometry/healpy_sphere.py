import sys
import numpy as np
from scipy.optimize import differential_evolution
from ..main import dataclass

from ..path_config import config
sys.path.insert(0, config["user_healpy_path"])
import healpy as hp

def healpy3Dsphere(self, 
                   variables,
                   r = 50, 
                   shell_pct = 0.05, 
                   weights = [None], 
                   max_unpopulated_pct = 0.1,
                   verbose = 1):
    
    try: self.trans_xyz
    except: self.calc_trans_xyz()

    shell_r = r / self.code2au
    Δ_r = np.maximum(shell_pct * shell_r, 2.5 * 0.5 ** self.lmax) 


    #### Set up dictonaries to handle both weigthed averages in overpopulated pixels and extraction of several variables #### 
    
    mask = (self.dist < shell_r + Δ_r) & (self.dist > shell_r - Δ_r)
    
    values = {ivs: [] for ivs in variables}
    for i, ivs in enumerate(variables):
        if (ivs == np.array(['d', 'P'])).any():
            values[ivs] = self.mhd[ivs][mask]
        else:
            values[ivs] = getattr(self, ivs)[mask]

    weights_dict = {ivs: [] for ivs in variables}
    for i, ivs in enumerate(variables):
        w = weights[i]
        if w != None:
            if w == 'mass': weights_dict[ivs] = self.m[mask]
            if w == 'volume': weights_dict[ivs] = (self.amr['ds']**3)[mask]
        else: 
            weights_dict[ivs] = np.ones(mask.sum())

    cartcoor = self.trans_xyz[:,mask]
    cell_size = self.amr['ds'][mask]

    #### Starting to assign cell values to pixels, but first the resolution of the pixels most be found ####
    x0 = differential_evolution(lambda x: abs(hp.nside2resol(x) * shell_r - cell_size.mean()), bounds = [(0, 500)])
    nside = int(np.rint(x0.x))

    no_coverage = 1
    while no_coverage > max_unpopulated_pct:
        npix = hp.nside2npix(nside); 
        pixel_indices = hp.vec2pix(nside, cartcoor[0], cartcoor[1], cartcoor[2])
        index, counts = np.unique(pixel_indices, return_counts=True); 
        m = np.zeros(npix)
        m[index] = counts
        no_coverage = np.sum(m == 0) / npix
        nside = np.rint((1 - no_coverage)**0.5 *nside).astype('int')
    
    npix = hp.nside2npix(nside); 
    pixel_indices = hp.vec2pix(nside, cartcoor[0], cartcoor[1], cartcoor[2])
    index, counts = np.unique(pixel_indices, return_counts=True)
    
    if verbose > 0:
        print('Number of pixels on the sphere: ',npix)
        print('Pixels without any representation: ', np.sum(m == 0))
        print(f'Percentage of no-coverage: {no_coverage * 100:2.2f} %' )


    cell_area = hp.nside2pixarea(nside) * shell_r**2
    maps = {ivs: [] for ivs in variables}    
    for i, ivs in enumerate(variables):
        v = values[ivs]
        w = weights_dict[ivs]
        pixel_sum = np.bincount(pixel_indices, v * w, minlength = npix)
        pixel_weight = np.bincount(pixel_indices, w, minlength = npix)
        map_clean = np.zeros(npix)
        map_clean[pixel_weight > 0] = pixel_sum[pixel_weight > 0] / pixel_weight[pixel_weight > 0]

        #### Starting interpolation for unpopulated pixels ###
        map_inter = map_clean.copy()
        pixel_i = np.array(np.where(map_inter == 0)).squeeze()
        all_neighbours = hp.get_all_neighbours(nside, pixel_i).squeeze()

        if verbose > 0:
            print('Interpolating unpopulated cells...')

        for i, index in enumerate(pixel_i):
            non_zero_neighbours = map_inter[all_neighbours[:,i]] != 0
            map_inter[index] = np.average(map_inter[all_neighbours[:,i]], weights = non_zero_neighbours)
        
        maps[ivs] = map_inter * cell_area

    return maps, nside

dataclass.healpy3Dsphere = healpy3Dsphere