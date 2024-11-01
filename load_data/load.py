import numpy as np
import tqdm, sys, os
from ..path_config import config 
from .polytrope import calc_pressure, calc_gamma
from ..main import dataclass
import tqdm

def load_RAMSES(snap, path):
    sys.path.insert(0, config["user_osyris_path"])
    import osyris
    # constants
    class units_class():
        def __init__(self):
            self.au = 14959787070000                  # 1 au in cm
            self.pc = self.au * 3600. * 180. / np.pi  # 1 parsec in cm
            self.yr = 3600. * 24. * 365.25            # astronomical year in seconds
            self.msun = 1.98847e33                    # solar masses in g
            
            # cgs units
            self.lcgs = 4. * self.pc                  # 4 parsec in cm
            self.vcgs = 1.8e4                         # 0.18 km/s
            self.tcgs = self.lcgs / self.vcgs         # 21.7 Myr in seconds
            
            # natural units
            self.l = self.lcgs / self.au              # box size in AU
            self.v = 0.18                             # velocity in km/s
            self.t = self.tcgs / self.yr              # time in yr
            self.n = 798.                             # number density in ppc
            self.m = 2998.                            # mass in solar masses
            self.d = (self.m * self.msun) / self.lcgs**3 # density in g / cm^-3

    unit = units_class()
    ds = osyris.Dataset(snap, path=path)
    ds.meta['unit_l'] = unit.lcgs
    ds.meta['unit_t'] = unit.tcgs
    ds.meta['unit_d'] = unit.d
    ds.set_units()
    data = ds.load()
    data['hydro']['gamma'] = osyris.Array(calc_gamma(data['hydro']['density']._array ))

    return data, ds

def load_DISPATCH(snap, sink_id, path, loading_bar):
    dict_amr = {key: [] for key in ['pos', 'ds']}
    dict_mhd = {key: [] for key in ['vel', 'B', 'd', 'P', 'm', 'gamma']}
    dict_sink = {key: [] for key in ['pos', 'vel', 'age', 'mass']}

    sys.path.insert(0, config["user_dispatch_path"])
    import dispatch as dis

    sn = dis.snapshot(snap, '.', data = path)

    #Sort the patces according to their level
    pp = [p for p in sn.patches]
    w = np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    for p in tqdm.tqdm(sorted_patches, disable = not loading_bar, desc = 'Loading patches'):
        p.m = p.var('d') * np.prod(p.ds)
        p.P = calc_pressure(p.var('d'))
        p.γ = calc_gamma(p.var('d'))
        p.xyz = np.array(np.meshgrid(p.xi, p.yi, p.zi, indexing='ij'))
        p.vel_xyz = np.concatenate([p.var(f'u'+axis)[None,...] for axis in ['x','y','z']], axis = 0)
        p.B =  np.concatenate([p.var(f'b'+axis)[None,...] for axis in ['x','y','z']], axis = 0)

        nbors = [sn.patchid[i] for i in p.nbor_ids if i in sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]
        if len(leafs) == 8: continue
        
        to_extract = np.ones(pp[0].n, dtype=bool)
        for lp in leafs: 
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) 
                                   & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool 
        
        dict_amr['pos'].extend((p.xyz[:,to_extract].T).tolist())
        dict_amr['ds'].extend((p.ds[0] * np.ones(to_extract.sum())))

        dict_mhd['vel'].extend((p.vel_xyz[:,to_extract].T).tolist())
        dict_mhd['B'].extend((p.B[:,to_extract].T).tolist())
        dict_mhd['d'].extend((p.var('d')[to_extract].T).tolist())
        dict_mhd['P'].extend((p.P[to_extract].T).tolist())
        dict_mhd['m'].extend((p.m[to_extract].T).tolist())     
        dict_mhd['gamma'].extend((p.γ[to_extract].T).tolist())
    
    #Load in sink data closest to the snapshot time
    sn_times = np.array([sink_out.time for sink_out in sn.sinks[sink_id]])
    sn_i = np.argmin(abs(sn.time - sn_times))
    dict_sink['pos'] = sn.sinks[sink_id][sn_i].position
    dict_sink['vel'] = sn.sinks[sink_id][sn_i].velocity
    dict_sink['age'] = sn.sinks[sink_id][sn_i].time
    dict_sink['mass'] = sn.sinks[sink_id][sn_i].mass

    for key in dict_amr:
        dict_amr[key] = np.array(dict_amr[key]).T
    for key in dict_mhd:
        dict_mhd[key] = np.array(dict_mhd[key]).T
    
    return dict_amr, dict_mhd, dict_sink

    
