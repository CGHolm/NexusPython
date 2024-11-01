import numpy as np
import sys, os, tqdm
from .path_config import config
import gc

class HiddenPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout

calc_ang = lambda vector1, vector2: np.rad2deg(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))

#### All data variables are saved in cgs units. (Execpt sink_mass which is in [Msun]] #####
class dataclass:
    def __init__(self, loading_bar = True):
        self.au = 14959787070000                         # 1 au in cm
        self.pc = self.au * 3600. * 180. / np.pi         # 1 parsec in cm
        self.yr = 3600. * 24. * 365.25                   # astronomical year in seconds
        self.msun = 1.98847e33                           # solar masses in g
        self.l_cgs = 4. * self.pc                         # 4 parsec in cm
        self.code2au = self.l_cgs / self.au
        self.v_cgs = 1.8e4                                # 0.18 km/s
        self.t_cgs = self.l_cgs / self.v_cgs                # 21.7 Myr in seconds  
        self.code2yr = self.t_cgs / self.yr                                   
        self.m_cgs = 2998 * self.msun
        self.code2msun = self.m_cgs / self.msun
        self.d_cgs = self.m_cgs / self.l_cgs**3             # density in g / cm^-3
        self.Σ_cgs = self.d_cgs * self.l_cgs
        self.P_cgs = self.d_cgs * (self.l_cgs / self.t_cgs)**2
        self.B_cgs = np.sqrt(4.0 * np.pi * self.d_cgs * self.v_cgs ** 2)
        
        self.loading_bar = loading_bar


        self.amr = {}
        self.mhd = {}
        
        # dtype = float64 for the Osryis implementation
    def load(self, io, snap, path, sink_id, verbose = 1, dtype = 'float32'):
        self.dtype = dtype
        self.io = io
        self.sink_id = sink_id
        if self.io == 'RAMSES':

            ####_______________________________LOADING RAMSES DATA____________________________________####

            from .load_data.load import load_RAMSES # type: ignore
            sys.path.insert(0, config["user_pyramses_path"])
            from pyramses.sink import rsink # type: ignore
            with HiddenPrints(suppress= not self.loading_bar ): 
                data, ds = load_RAMSES(snap = snap, path = path)
        
            self.amr['pos'] = np.asarray([getattr(data['amr']['position'], coor)._array / self.l_cgs - 0.5 for coor in ['x', 'y', 'z']], dtype = self.dtype)
            self.amr['ds'] = np.array(data['amr']['dx']._array / self.l_cgs, dtype=self.dtype).squeeze()

            #Osyris reads in the data but assigns cgs units.
            # Nexus reverts this so both Dispatch and RAMSES data will be handled in code units 
            # to avoid excessive memery usage and overflow problems
            for save, read, unit in zip(['vel', 'B'], ['velocity', 'B_field'], ['v_cgs', 'B_cgs']):
                 self.mhd[save] = np.asarray([getattr(data['hydro'][read], coor)._array / getattr(self, unit) for coor in ['x', 'y', 'z']], dtype = self.dtype)

            for save, read, unit in zip(['d', 'P', 'm'], ['density', 'thermal_pressure','mass'], ['d_cgs', 'P_cgs', 'code2msun']):
                self.mhd[save] = np.array(data['hydro'][read]._array / getattr(self, unit), dtype = self.dtype).squeeze()
            self.mhd['gamma'] = np.array(data['hydro']['gamma']._array, dtype = self.dtype).squeeze() 

            del data
            s=rsink(snap, datadir=path, sink_id=self.sink_id)
            self.sink_pos = (np.array([s[coor][self.sink_id] for coor in ['x','y','z']], dtype = self.dtype) - 0.5)
            self.sink_vel = np.array([s[v_comp][self.sink_id] for v_comp in ['ux', 'uy', 'uz']], dtype = self.dtype) 
            self.time = s['snapshot_time']
            self.sink_mass = s['m'][self.sink_id] 
        
        if self.io == 'DISPATCH':
            
            ####_______________________________LOADING DISPATCH DATA____________________________________####
            
            from .load_data.load import load_DISPATCH
            dict_amr, dict_mhd, dict_sink = load_DISPATCH(snap, self.sink_id, path, self.loading_bar)
            
            for key in dict_amr:
                self.amr[key] = dict_amr[key].astype(self.dtype)
            
            for key in dict_mhd:
                self.mhd[key] = dict_mhd[key].astype(self.dtype) 

            self.sink_pos = dict_sink['pos'].astype(self.dtype)
            self.sink_vel = dict_sink['vel'].astype(self.dtype) 
            self.time = dict_sink['age'].astype(self.dtype) 
            self.sink_mass = dict_sink['mass'].astype(self.dtype) 

        assert (self.amr['pos'].min() > -0.5) & (self.amr['pos'].max() < 0.5), 'Data snapshot might be corrupted'     

        gc.collect() # Clean memory

        if verbose > 0: print('Assigning relative coordinates to all 1D vectors...')
        self.lmax = int(np.log(self.amr['ds'].min()) / np.log(0.5))
        self.rel_xyz = (self.amr['pos'] - self.sink_pos[:,None]).astype(self.dtype)
        self.rel_xyz[self.rel_xyz < - 0.5] += 1
        self.rel_xyz[self.rel_xyz > 0.5 ] -= 1
        self.vrel = (self.mhd['vel'] - self.sink_vel[:,None]).astype(self.dtype)
        self.dist = np.linalg.norm(self.rel_xyz, axis = 0).astype(self.dtype)
        self.m = self.mhd['m'].astype(self.dtype)

    # Get the angular momentum from a sphere with radius r in au
    def calc_Lshere(self, r = 50):
        mask = self.dist < r / self.code2au
        L = np.sum(np.cross(self.rel_xyz[:,mask], self.vrel[:,mask] * self.m[mask], axisa=0, axisb=0, axisc=0), axis = 1)
        self.L = L / np.linalg.norm(L)

    def define_cyl(self):
        try: self.L
        except: self.calc_Lshere()
        self.cyl_z = np.sum(self.L[:,None] * self.rel_xyz, axis = 0).astype(self.dtype)
        self.cyl_r = self.rel_xyz -  self.cyl_z * self.L[:,None]     
        self.cyl_R = np.linalg.norm(self.cyl_r, axis = 0).astype(self.dtype)
        self.e_r = self.cyl_r / self.cyl_R
        self.e_phi = np.cross(self.L, self.e_r, axisa=0, axisb=0, axisc=0).astype(self.dtype)
        
    def recalc_L(self, h = 15, r = 150, aspect_ratio = 0.3, err_deg = 5, verbose = 1):
        h /= self.code2au; r /= self.code2au
        try: self.cyl_z
        except: self.define_cyl()
        def reclac():
            mask = (self.cyl_R < r) & ((abs(self.cyl_z) < h) | (abs(self.cyl_z / self.cyl_R) < aspect_ratio))
            L = np.sum(np.cross(self.rel_xyz[:,mask], self.vrel[:,mask] * self.m[mask], axisa=0, axisb=0, axisc=0), axis = 1)
            return L/np.linalg.norm(L)
        L_new = reclac()
        L_iter = 0
        while calc_ang(self.L, L_new) > err_deg:
            self.L = L_new
            L_new = reclac()
            L_iter += 1
            if L_iter > 10: break
        if verbose != 0: print(f'Converged mean angular momentum vector after {L_iter} iteration(s)')


    def calc_trans_xyz(self, verbose = 1, top = 'L'):
        try: self.e_r
        except: self.recalc_L()

        #https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
        def rotation_matrix_func(axis, theta):
            """
            Return the rotation matrix associated with counterclockwise rotation about
            the given axis by theta radians.
            """
            axis = np.asarray(axis)
            axis = axis / np.sqrt(np.dot(axis, axis))
            a = np.cos(theta / 2.0)
            b, c, d = -axis * np.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        rotation_axis = np.cross(np.array([0, 0, 1]), self.L)
        theta = np.arccos(np.clip(np.dot(np.array([0, 0, 1]), self.L), -1.0, 1.0))
        rotation_matrix = rotation_matrix_func(rotation_axis, theta)
        self.rotation_matrix = rotation_matrix

        if verbose > 0:
            print('Transforming old z-coordinate into mean angular momentum vector')
        self.new_x = np.dot(self.rotation_matrix, np.array([1,0,0])) 
        self.new_y = np.dot(self.rotation_matrix, np.array([0,1,0]))
        self.L = np.dot(self.rotation_matrix, np.array([0, 0, 1]))

        if top != 'L':
            if top == 'x':
                new_x = self.new_y.copy()
                new_y = self.L.copy()
                new_L = self.new_x.copy()
            elif top == 'y':
                new_x = self.L.copy()
                new_y = self.new_x.copy()
                new_L = self.new_y.copy()
            self.new_x = new_x; self.new_y = new_y; self.L = new_L

        self.trans_xyz = np.array([np.sum(coor[:, None] * self.rel_xyz, axis=0) for coor in [self.new_x, self.new_y, self.L]]).astype(self.dtype)
        self.trans_vrel = np.array([np.sum(coor[:, None] * self.vrel, axis=0) for coor in [self.new_x, self.new_y, self.L]]).astype(self.dtype)
        proj_r = np.sum(self.cyl_r * self.new_x[:, None], axis=0).astype(self.dtype)
        proj_φ = np.sum(self.cyl_r * self.new_y[:, None], axis=0).astype(self.dtype)
        self.φ = np.arctan2(proj_φ, proj_r) + np.pi
    


        

        
        