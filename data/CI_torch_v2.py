import numpy as np
import ehtim as eh
from astropy.time import Time
import torch
import pandas as pd

from ClosureInvariants import graphUtils as GU
from ClosureInvariants import scalarInvariants_torch as SI
from ClosureInvariants import vectorInvariants_torch as VI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Closure_Invariants():

    def __init__(self, ehtarray='EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9], psize=1.7044214966184275e-11,
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 uvfits_files=None):


        array = eh.array.load_txt(ehtarray)
        if subarray is not None:
            # array = array.make_subarray(['ALMA','APEX','LMT','PV','SMT','JCMT','SMA'])
            array = array.make_subarray(subarray)
        
        t = Time(date, format='iso', scale='utc')

        self.fovx = psize*206265*1e6 * 64 # imgdim
        self.fovy = psize*206265*1e6 * 64 # imgdim
        
        self.obslist = []
        if uvfits_files is not None:
            for uvfits in uvfits_files:
                obs = eh.obsdata.load_uvfits(uvfits)
                self.obslist.append(obs)
        else:
            for rf in bw_hz:
                obs = array.obsdata(ra/360*24, dec, rf, 8e9, tint_sec, tadv_sec, tstart_hr, tstop_hr, mjd=int(t.mjd), timetype='UTC')
                self.obslist.append(obs)

        uvwlist = []
        antenna_list = []
        num_site_pairs = []
        site_pairs = []
        timestamps = []
        sigmas = []
        obs_vislist = []
        for obs in self.obslist:
            for tdata in obs.tlist():
                num_antenna = len(np.unique(tdata['t1'])) + 1
                if num_antenna < 3:
                    continue
                u, v = tdata['u'], tdata['v']
                try:
                    w = tdata['w']
                except:
                    w = np.zeros_like(u)
                antenna_list.append(num_antenna)
                timestamps.append(tdata['time'])
                uvwlist.append(np.stack((u, v, w), axis=-1))
                site_pairs.append(self.recarr_to_ndarr(tdata[['t1', 't2']], 'U32'))
                num_site_pairs.append(len(site_pairs[-1]))
                sigmas.append(tdata['sigma'])
                obs_vislist.append(tdata['vis'])
        
        # sort_idx = np.argsort(antenna_list, kind='stable')
        sort_idx = np.argsort(num_site_pairs, kind='stable')
        uvwlist = [uvwlist[i] for i in sort_idx]
        site_pairs = [site_pairs[i] for i in sort_idx]
        timestamps = [timestamps[i] for i in sort_idx]
        sigmas = [sigmas[i] for i in sort_idx]
        obs_vislist = [obs_vislist[i] for i in sort_idx]

        uvwlist = np.concatenate(uvwlist, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)
        
        # group by number of site_pairs
        N_site_pairs = np.array([len(i) for i in site_pairs])
        unique_N_site_pairs = np.unique(N_site_pairs)
        self.N_times = [int(np.sum(N_site_pairs == i)) for i in unique_N_site_pairs]
        self.N_idx = [int(np.sum(N_site_pairs == i)*i) for i in unique_N_site_pairs]
        
        # split site_pairs by times
        site_ids_flat = np.concatenate(site_pairs, axis=0)
        site_pairs = np.array(site_pairs, dtype=object)
        site_pairs = np.split(site_pairs, np.cumsum(self.N_times)[:-1])

        sigmas = np.array(sigmas, dtype=object)
        sigmas = np.concatenate(sigmas, axis=0)

        obs_vislist = np.array(obs_vislist, dtype=object)
        obs_vislist = np.concatenate(obs_vislist, axis=0)

        self.site_pairs = site_pairs
        self.site_ids_flat = site_ids_flat
        self.sigmas = sigmas
        self.obs_vislist = obs_vislist
        self.uvwlist = uvwlist
        self.timestamps = timestamps
        self.bs = 1


    def FTCI_batch(self, batch, imgs, add_th_noise=False,  th_noise_factor=1,
                   return_combined=False,
                   fov=None, fovx=None, fovy=None, intensity=0, stokes=False):
        # split images into batches
        imgs_batched = np.array_split(imgs, batch)
        for i, img in enumerate(imgs_batched):
            ci = self.FTCI(img, add_th_noise=add_th_noise, return_uv=False, return_combined=return_combined, th_noise_factor=th_noise_factor,
                           fov=fov, fovx=fovx, fovy=fovy, intensity=intensity, stokes=stokes)
            if i == 0:
                ci_batch = ci
            else:
                ci_batch = torch.concatenate((ci_batch, ci), dim=0)
        return ci_batch


    def FTCI(self, imgs, add_th_noise=False, th_noise_factor=1,
             return_uv=False, return_vis=False, return_combined=False, return_list=False,
             fov=None, fovx=None, fovy=None, intensity=0, stokes=False,
             useObs=False):
        
        if stokes and imgs.shape[1] != 4:
            raise ValueError("Stokes parameter requires 4 channels in the image")
        elif not stokes and imgs.shape[1] == 4:
            imgs = imgs[:, 0, :, :]

        self.bs = imgs.shape[0]
        if intensity > 0:
            imgs = imgs * intensity

        if isinstance(imgs, np.ndarray):
            imgs = torch.tensor(imgs).to(device)
            
        ci = torch.tensor(torch.empty((imgs.shape[0], 0)))
        out_uv = []

        if fov is not None:
            fovx = fov
            fovy = fov
        elif fovx is None or fovy is None:
            fovx = self.fovx
            fovy = self.fovy

        vis = self.Visibilities(imgs, torch.tensor(self.uvwlist[:,:2], dtype=torch.float32).to(device), fovx, fovy, stokes=False)
        if useObs:
            vis = torch.tensor(self.obs_vislist, dtype=torch.complex128).to(device).reshape(1, -1)

        if stokes:
            vis = vis.reshape((len(imgs), 4, -1))
            vis = self.stokes_to_Bmatrix(vis)
        
        if add_th_noise:
            vis = self.add_noise(vis, self.sigmas, th_noise_factor=th_noise_factor, stokes=stokes)

        vis_by_time = torch.split(vis, self.N_idx, dim=1)
        vis_by_time = [i.clone() for i in vis_by_time]
        for ind, (i, times) in enumerate(zip(vis_by_time, self.N_times)):
            if stokes:
                vis_by_time[ind] = i.reshape([self.bs, times] + [-1, 2, 2])
            else:
                vis_by_time[ind] = i.reshape([self.bs, times] + [-1])

        time_by_time = torch.split(torch.tensor(self.timestamps, dtype=torch.float32), self.N_idx, dim=0)
        time_by_time = [i.clone() for i in time_by_time]
        for ind, (i, times) in enumerate(zip(time_by_time, self.N_times)):
            time_by_time[ind] = i.reshape(times, -1)

        uv_by_time = torch.split(torch.tensor(self.uvwlist, dtype=torch.float32), self.N_idx, dim=0)
        uv_by_time = [i.clone() for i in uv_by_time]

        for ind, (i, times) in enumerate(zip(uv_by_time, self.N_times)):
            uv_by_time[ind] = i.reshape(times, -1, 3)
    
        out_list = []
        for ind, (temp_vis, uv, pairs, time) in enumerate(zip(vis_by_time, uv_by_time, self.site_pairs, time_by_time)):
            if return_uv or return_list:
                temp_ci, temp_uv = self.ClosureInvariants(temp_vis, uv=uv, pairs=pairs[0], stokes=stokes)
                out_uv.append(temp_uv)
            else:
                temp_ci, _ = self.ClosureInvariants(temp_vis, uv=None, pairs=pairs[0], stokes=stokes)
            
            if return_list:
                time = time[:,0]
                element_pairs = np.array([pd.unique(i.ravel()) for i in pairs])
                temp_uv = temp_uv.reshape(1, 3, 3, -1, temp_ci.shape[-1])
                temp_uv = temp_uv[0]
                out_list.append([time, element_pairs, temp_uv, temp_ci.cpu().detach().numpy()])

            temp_ci = temp_ci.reshape(imgs.shape[0], -1)
            ci = torch.cat((ci, temp_ci), dim=1)

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
    

    def ClosureInvariants(self, vis, uv=None, pairs=None, stokes=False):
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
        triads_indep = GU.generate_independent_triads(element_ids, baseid=element_ids[0])
        if stokes:
            pol_axes = (-2, -1)
            bl_axis = -3
            vis = vis.to(torch.complex128)
            corrs_lol = VI.corrs_list_on_loops(vis.cpu(), element_pairs, triads_indep, bl_axis=bl_axis, pol_axes=pol_axes)
            advariants = VI.advariants_multiple_loops(corrs_lol, pol_axes=pol_axes)
            if advariants.dim() == 3:
                advariants = advariants.unsqueeze(1)
            z4 = VI.vector_from_advariant(advariants)
            mdp = VI.complete_minkowski_dots(z4)
            ci = VI.remove_scaling_factor_minkoski_dots(mdp, wts=None)
        else:
            corrs_lol = SI.corrs_list_on_loops(vis.cpu(), element_pairs, triads_indep, bl_axis=-1)
            advariants = SI.advariants_multiple_loops(corrs_lol)
            ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts=None, normpower=2)
            # ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts='max', normpower=1)

        if uv is not None: 
            uv = uv.swapaxes(1, 2)
            triads_indep = np.array(triads_indep)
            triads_indep = [np.where(np.all(np.sort(np.array(element_pairs)) == np.sort([triad[i], triad[i+1]]), axis=1))[0][0] for triad in triads_indep for i in (-1, 0, 1)]
            triads_indep = np.array(triads_indep).reshape(-1, 3)
            # print(triads_indep.shape)
            uv0 = uv[:, :, np.array(triads_indep)[:, 0]]
            uv1 = uv[:, :, np.array(triads_indep)[:, 1]]
            uv2 = uv[:, :, np.array(triads_indep)[:, 2]]
            uv = np.dstack((uv0,  uv1,  uv2))
            uv = uv.T.swapaxes(0, 1)
            if stokes:
                uv = uv.reshape(1, 3, 3, advariants.shape[-3], -1)
                if advariants.shape[-3] > 2:
                    basis = uv[:, :, :, :2, :].repeat(5, axis=-2)
                    rest = uv[:, :, :, 2:, :].repeat(8, axis=-2)
                    uv = np.concatenate((basis, rest), axis=-2)
                else:
                    uv = uv.repeat(mdp.shape[-1], axis=-2)
            else:
                uv = uv.reshape(1, 3, 3, advariants.shape[-1], -1)
                uv = np.concatenate((uv, uv), axis=-1)
            uv = uv.reshape(1, 3, 3, -1)
            return ci, uv
        
        return ci, None 

    def add_noise(self, vis, sigmas, th_noise_factor=1, stokes=False):
        sigmas = torch.tensor(sigmas).to(device) * th_noise_factor
        if stokes:
            sigmas = sigmas[:, None, None]
            sigmas = sigmas.repeat(1, 2, 2).to(device)
        vis_noise = vis + ((torch.randn_like(vis) + 1j*torch.randn_like(vis))* sigmas)
        return vis_noise



############################################################################################################
 

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


    def Visibilities(self, imgs, uv=None, xfov=225, yfov=225, stokes=False):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (torch.Tensor): tensor of images

        Returns:
            vis (torch.Tensor): visibilities taken for each image
        """
        vis = self.DFT(imgs, uv, xfov, yfov)
        if stokes:
            vis = vis.reshape((len(imgs), 4, -1))
            return self.stokes_to_Bmatrix(vis)
        else:
            return vis.reshape((len(imgs), -1))
        
    def stokes_to_Bmatrix(self, stokes_vis):
        I, Q, U, V = stokes_vis[:, 0], stokes_vis[:, 1], stokes_vis[:, 2], stokes_vis[:, 3]
        B = torch.zeros((2, 2, len(stokes_vis), stokes_vis.shape[-1]), dtype=I.dtype)
        B[0, 0] = I + Q
        B[0, 1] = U + 1j*V
        B[1, 0] = U - 1j*V
        B[1, 1] = I - Q
        B = B.permute(2, 3, 0, 1)
        return B.to(device)
    
    def recarr_to_ndarr(self, x, typ):
        """converts a record array x to a normal ndarray with all fields converted to datatype typ
        """
        fields = x.dtype.names
        shape = x.shape + (len(fields),)
        dt = [(name, typ) for name in fields]
        y = x.astype(dt).view(typ).reshape(shape)
        return y