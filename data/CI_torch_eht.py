import numpy as np
import ehtim as eh
from astropy.time import Time
import torch
from ehtim.observing import obs_helpers as obsh
from ehtim.observing import obs_simulate as obss
import pandas as pd

from ClosureInvariants import graphUtils as GU
from ClosureInvariants import scalarInvariants as SI
from ClosureInvariants import vectorInvariants as VI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Closure_Invariants():

    def __init__(self, ehtarray='EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9], psize=7.757018897750619e-12,
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 noise=False, sgrscat=False, ampcal=True, phasecal=True,
                 reorder=True):


        t = Time(date, format='iso', scale='utc')
        self.mjd = int(t.mjd)
        self.ra = ra/360*24
        self.dec = dec
        self.bw_hz = bw_hz
        self.tint_sec = tint_sec
        self.tadv_sec = tadv_sec
        self.tstart_hr = tstart_hr
        self.tstop_hr = tstop_hr
        self.noise = noise
        self.sgrscat = sgrscat
        self.ampcal = ampcal
        self.phasecal = phasecal
        self.psize = psize

        self.ehtarray = eh.array.load_txt(ehtarray)
        if subarray is not None:
            self.ehtarray = self.ehtarray.make_subarray(['ALMA','APEX','LMT','PV','SMT','JCMT','SMA'])
        
        template = './data/template_sgra.txt'
        template = eh.image.load_txt(template)
        template.ra = self.ra
        template.dec = self.dec
        template.mjd = self.mjd
        template.psize = self.psize * 64/100 # because template_sgra has 100 pixels rather than 64, OOPS
        self.template = template.regrid_image(template.fovx(), 64, 'cubic')
        self.fovx = self.template.fovx()*206265*1e6
        self.fovy = self.template.fovy()*206265*1e6
        
        self.obslist = []
        for bw in self.bw_hz:
            self.template.rf = bw # actually the radio frequency not bandwidth
            obs = self.template.observe(self.ehtarray, self.tint_sec, self.tadv_sec, self.tstart_hr, self.tstop_hr, 8e9, # 8 GHz bandwidth
                                        mjd = self.mjd, timetype='UTC', ttype='DFT', noise=False, reorder=reorder, verbose=False)
            self.obslist.append(obs)

        uvlist = []
        uvwlist = []
        uv_complex = []
        antenna_list = []
        site_pairs = []
        obs_vis_list = []
        timestamps = []
        for obs in self.obslist:
            for tdata in obs.tlist():
                num_antenna = len(np.unique(tdata['t1'])) + 1
            
                if num_antenna < 3:
                    continue

                antenna_list.append(num_antenna)
                u = tdata['u']
                v = tdata['v']
                w = tdata['w']

                timestamps.append(tdata['time'])
                uvlist.append(np.stack((u, v), axis=-1))
                uvwlist.append(np.stack((u, v, w), axis=-1))
                uv_complex.append(u + 1j*v)

                obs_vis_list.append(tdata['vis'])

                sites = obsh.recarr_to_ndarr(tdata[['t1', 't2']], 'U32')
                site_pairs.append(sites)
        
        sort_idx = np.argsort(antenna_list)
        uvlist = [uvlist[i] for i in sort_idx]
        uvwlist = [uvwlist[i] for i in sort_idx]
        uv_complex = [uv_complex[i] for i in sort_idx]
        obs_vis_list = [obs_vis_list[i] for i in sort_idx]
        site_pairs = [site_pairs[i] for i in sort_idx]
        timestamps = [timestamps[i] for i in sort_idx]

        uvlist = np.concatenate(uvlist, axis=0)
        uvwlist = np.concatenate(uvwlist, axis=0)
        uv_complex = np.concatenate(uv_complex, axis=0)
        obs_vis_list = np.concatenate(obs_vis_list, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)
        
        # group by number of site_pairs
        N_site_pairs = np.array([len(i) for i in site_pairs])
        unique_N_site_pairs = np.unique(N_site_pairs)
        N_times = [int(np.sum(N_site_pairs == i)) for i in unique_N_site_pairs]
        N_idx = [int(np.sum(N_site_pairs == i)*i) for i in unique_N_site_pairs]
        
        # split site_pairs by times
        site_pairs = np.array(site_pairs, dtype=object)
        site_pairs = np.split(site_pairs, np.cumsum(N_times)[:-1])

        self.N_times = N_times
        self.N_idx = N_idx

        self.site_pairs = site_pairs
        self.uvlist = uvlist
        self.uvwlist = uvwlist
        self.uv_complex = uv_complex
        self.unique_N_site_pairs = unique_N_site_pairs
        self.timestamps = timestamps


        obs_idx = []
        for obs in self.obslist:
            idx = [np.where(uv == uv_complex)[0] for uv in obs.data['u']+1j*obs.data['v']]
            obs_idx.append(idx)
        

        non_empty_idx = []
        for idx in obs_idx:
            non_empty_idx.append(np.where(np.array([len(i) for i in idx]) != 0)[0])
        self.non_empty_idx = non_empty_idx
        self.obs_idx = [np.concatenate(i) for i in obs_idx]

        self.bs = 1


    def FTCI_batch(self, batch, imgs, add_th_noise=False,  th_noise_factor=1,
                   return_combined=False,
                   fov=None, fovx=None, fovy=None, intensity=0):
        # split images into batches
        imgs_batched = np.array_split(imgs, batch)
        for i, img in enumerate(imgs_batched):
            ci = self.FTCI(img, add_th_noise=add_th_noise, return_uv=False, return_combined=return_combined, th_noise_factor=th_noise_factor,
                           fov=fov, fovx=fovx, fovy=fovy, intensity=intensity)
            if i == 0:
                ci_batch = ci
            else:
                ci_batch = np.concatenate((ci_batch, ci), axis=0)
        return ci_batch


    def FTCI(self, imgs, add_th_noise=False, th_noise_factor=1,
             return_uv=False, return_vis=False, return_combined=False, return_list=False,
             fov=None, fovx=None, fovy=None, intensity=0, normpower=2):
        self.bs = imgs.shape[0]
        if intensity > 0:
            imgs = imgs * intensity

        if isinstance(imgs, np.ndarray):
            imgs = torch.tensor(imgs).to(device)
            
        ci = np.array([np.array([]) for i in range(len(imgs))])
        out_uv = []

        if fov is not None:
            fovx = fov
            fovy = fov
        elif fovx is None or fovy is None:
            fovx = self.fovx
            fovy = self.fovy

        vis = self.Visibilities(imgs, torch.tensor(self.uvlist, dtype=torch.float32).to(device), fovx, fovy)
        if add_th_noise:
            for idx, not_empty, obs in zip(self.obs_idx, self.non_empty_idx, self.obslist):
                vis = self.add_ehtim_noise(obs, vis, idx, not_empty, add_th_noise=add_th_noise, th_noise_factor=th_noise_factor)

        vis_by_time = torch.split(vis, self.N_idx, dim=1)
        vis_by_time = [i.detach().cpu().numpy() for i in vis_by_time]
        # vis_by_time = [torch.tensor(i) for i in vis_by_time]
        for ind, (i, times) in enumerate(zip(vis_by_time, self.N_times)):
            vis_by_time[ind] = i.reshape(self.bs, times, -1)

        time_by_time = torch.split(torch.tensor(self.timestamps, dtype=torch.float32).to(device), self.N_idx, dim=0)
        time_by_time = [i.detach().cpu().numpy() for i in time_by_time]
        # time_by_time = [torch.tensor(i) for i in time_by_time]
        for ind, (i, times) in enumerate(zip(time_by_time, self.N_times)):
            time_by_time[ind] = i.reshape(times, -1)

        uv_by_time = torch.split(torch.tensor(self.uvwlist, dtype=torch.float32).to(device), self.N_idx, dim=0)
        uv_by_time = [i.detach().cpu().numpy() for i in uv_by_time]
        # uv_by_time = [torch.tensor(i) for i in uv_by_time]

        for ind, (i, times) in enumerate(zip(uv_by_time, self.N_times)):
            uv_by_time[ind] = i.reshape(times, -1, 3)
    
        out_list = []
        for ind, (temp_vis, uv, pairs, time) in enumerate(zip(vis_by_time, uv_by_time, self.site_pairs, time_by_time)):
            time = time[:,0]
            if return_uv:
                temp_ci, temp_uv = self.ClosureInvariants(torch.tensor(temp_vis), uv=uv, pairs=pairs[0], normpower=normpower)
                out_uv.append(temp_uv)
            else:
                temp_ci, _ = self.ClosureInvariants(torch.tensor(temp_vis), uv=None, pairs=pairs[0], normpower=normpower)
            
            if return_list:
                element_pairs = np.array([pd.unique(i.ravel()) for i in pairs])
                temp_uv = temp_uv.reshape(1, 3, 3, -1, temp_ci.shape[-1])
                temp_uv = temp_uv[0]
                out_list.append([time, element_pairs, temp_uv, temp_ci.cpu().detach().numpy()])

            temp_ci = temp_ci.reshape(imgs.shape[0], -1).cpu().detach().numpy()
            ci = np.concatenate((ci, temp_ci), axis=1)
        ci = torch.tensor(ci)

        if return_list:
            return out_list

        if return_combined:
            vis = torch.stack((vis.real, vis.imag), dim=1)
            vis = vis.reshape(vis.shape[0], -1).detach().cpu()
            ci = torch.cat((ci, vis), dim=1)

        if return_uv and return_vis:
            out_uv = np.concatenate(out_uv, axis=-1)
            return ci, vis, out_uv
        
        if return_uv:
            out_uv = np.concatenate(out_uv, axis=-1)
            return ci, out_uv
        
        if return_vis:
            return ci, vis
        
        return ci
    

    def Visibilities(self, imgs, uv=None, xfov=225, yfov=225):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (torch.Tensor): tensor of images

        Returns:
            vis (torch.Tensor): visibilities taken for each image
        """
        vis = self.DFT(imgs, uv, xfov, yfov)
        return vis.reshape((len(imgs), -1))


    def ClosureInvariants(self, vis, uv=None, pairs=None, normpower=2):
        """
        Calculates copolar closure invariants for visibilities assuming an n element 
        interferometer array using method 1.

        Nithyanandan, T., Rajaram, N., Joseph, S. 2022 “Invariants in copolar 
        interferometry: An Abelian gauge theory”, PHYS. REV. D 105, 043019. 
        https://doi.org/10.1103/PhysRevD.105.043019 

        Args:
            vis (torch.Tensor): visibility data sampled by the interferometer array
            n (int): number of antenna as part of the interferometer array

        Returns:
            ci (torch.Tensor): closure invariants
        """
        element_pairs = pairs
        element_ids = pd.unique(np.array(element_pairs).ravel())
        triads_indep = GU.generate_triangles(element_ids, baseid=element_ids[0])
        corrs_lol = SI.corrs_list_on_loops(vis.detach().cpu().numpy(), element_pairs, triads_indep, bl_axis=-1)
        advariants = SI.advariants_multiple_loops(corrs_lol)
        ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts=None, normpower=normpower)
        # ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts='max', normpower=1)
        ci = torch.tensor(ci)

        
        if uv is not None: 
            uv = uv.swapaxes(1, 2)
            triads_indep = np.array(triads_indep)
            triads_indep = [np.where(np.all(np.sort(np.array(element_pairs)) == np.sort([triad[i], triad[i+1]]), axis=1))[0][0] for triad in triads_indep for i in (-1, 0, 1)]
            triads_indep = np.array(triads_indep).reshape(-1, 3)
            uv0 = uv[:, :, np.array(triads_indep)[:, 0]]
            uv1 = uv[:, :, np.array(triads_indep)[:, 1]]
            uv2 = uv[:, :, np.array(triads_indep)[:, 2]]
            uv = np.dstack((uv0,  uv1,  uv2))
            uv = uv.T.swapaxes(0, 1)
            uv = uv.reshape(1, 3, 3, ci.shape[-2], -1)
            uv = np.concatenate((uv, uv), axis=-1)
            uv = uv.reshape(1, 3, 3, -1)
            return ci, uv
        
        return ci, None 
 

    def DFT(self, data, uv, xfov=225, yfov=225):
        if data.ndim == 2:
            data = data[None,...]
            out_shape = (uv.shape[0],)
        elif data.ndim > 2:
            data = data.reshape((-1,) + data.shape[-2:])
            out_shape = data.shape[:-2] + (uv.shape[0],)
        ny, nx = data.shape[-2:]
        dx = xfov*4.84813681109536e-12 / nx
        dy = yfov*4.84813681109536e-12 / ny
        angx = (torch.arange(nx) - nx//2) * dx
        angy = (torch.arange(ny) - ny//2) * dy
        lvect = torch.sin(angx)
        mvect = torch.sin(angy)
        l, m = torch.meshgrid(lvect, mvect)
        lm = torch.cat([l.reshape(1,-1), m.reshape(1,-1)], dim=0).to(device)
        imgvect = data.reshape((data.shape[0],-1)).to(device)
        x = -2*torch.pi*torch.matmul(uv,lm)[None, ...].to(device)
        visr = torch.sum(imgvect[:, None, :] * torch.cos(x).to(device), axis=-1)
        visi = torch.sum(imgvect[:, None, :] * torch.sin(x).to(device), axis=-1)
        if data.ndim == 2:
            vis = visr.ravel() + 1j*visi.ravel()
        else:
            vis = visr.ravel() + 1j*visi.ravel()
            vis = vis.reshape(out_shape)
        return vis

    def add_ehtim_noise(self, obs, vis, idx, not_empty, add_th_noise=True, th_noise_factor=1, opacitycal=True, ampcal=True, phasecal=True,
                        taup=0.1, gainp=0.1, gain_offset=0.1):
        # use ehtim to return vis with noise given original obs
        obsdata = obss.add_noise(obs, add_th_noise=add_th_noise, th_noise_factor=th_noise_factor, 
                                opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal,
                                taup=taup, gainp=gainp, gain_offset=gain_offset, verbose=False)
        delta_vis = (obsdata['vis'] - obs.data['vis'])[not_empty]
        vis[:, idx] += torch.tensor(delta_vis).to(device)
        return vis
        

    def Triads(self, n:int, pairs=None):
        """
        Generates arrays of antenna and baseline indicies that form triangular 
        loops pivoted around the 0th antenna. Used to calculate closure invariants
        whereby specific baseline correlations need to be indexed according 
        to those triangular loops.
        Baseline array format [ant1, ant2]:
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6] ... 
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6] ...
        [2, 3], [2, 4], [2, 5], [2, 6] ...
        [3, 4], [3, 5], [3, 6] ...
        [4, 5], [4, 6] ...
        [5, 6] ...

        Args:
            n (int): number of antenna in the array

        Returns:
            atriads (torch.Tensor): antenna triangular loop indicies
            btriads (torch.Tensor): baseline triangular loop indicies
        """
        ntriads = (n-1)*(n-2)//2
        ant1 = torch.zeros(ntriads, dtype=torch.uint8)
        ant2 = torch.arange(1, n, dtype=torch.uint8).reshape(n-1, 1) + torch.zeros(n-2, dtype=torch.uint8).reshape(1, n-2)
        ant3 = torch.arange(2, n, dtype=torch.uint8).reshape(1, n-2) + torch.zeros(n-1, dtype=torch.uint8).reshape(n-1, 1)
        anti = torch.where(ant3 > ant2)
        ant2, ant3 = ant2[anti], ant3[anti]
        atriads = torch.cat([ant1.reshape(-1, 1), ant2.reshape(-1, 1), ant3.reshape(-1, 1)], dim=-1)
        
        ant_pairs_01 = list(zip(ant1, ant2))
        ant_pairs_12 = list(zip(ant2, ant3))
        ant_pairs_20 = list(zip(ant3, ant1))
        
        t1 = torch.arange(n, dtype=int).reshape(n, 1) + torch.zeros(n, dtype=int).reshape(1, n)
        t2 = torch.arange(n, dtype=int).reshape(1, n) + torch.zeros(n, dtype=int).reshape(n, 1)
        bli = torch.where(t2 > t1)
        t1, t2 = t1[bli], t2[bli]
        if pairs == None:
            bl_pairs = list(zip(t1, t2))
        else:
            bl_pairs = pairs

        bl_01 = torch.tensor([bl_pairs.index(apair) for apair in ant_pairs_01])
        bl_12 = torch.tensor([bl_pairs.index(apair) for apair in ant_pairs_12])
        bl_20 = torch.tensor([bl_pairs.index(tuple(reversed(apair))) for apair in ant_pairs_20])
        btriads = torch.cat([bl_01.reshape(-1, 1), bl_12.reshape(-1, 1), bl_20.reshape(-1, 1)], dim=-1)
        return atriads, btriads

    def nanmax(tensor, dim=None, keepdim=False):
        min_value = torch.finfo(tensor.dtype).min
        output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
        return output

