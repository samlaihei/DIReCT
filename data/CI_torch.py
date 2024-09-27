import numpy as np
import ehtim as eh
from astropy.time import Time
import torch
from ehtim.observing import obs_helpers as obsh
import pandas as pd

from ClosureInvariants import graphUtils as GU
from ClosureInvariants import scalarInvariants as SI
from ClosureInvariants import vectorInvariants as VI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Closure_Invariants():

    def __init__(self, filename='ehtuv.npz', ehtim=False, ehtarray='EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9], psize=7.757018897750619e-12,
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 noise=False, sgrscat=False, ampcal=True, phasecal=True,
                 reorder=True):
        
        self.ehtim = ehtim

        if ehtim:
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
                template.rf = bw # actually the radio frequency not bandwidth
                obs = self.template.observe(self.ehtarray, self.tint_sec, self.tadv_sec, self.tstart_hr, self.tstop_hr, 8e9, # 8 GHz bandwidth
                                            mjd = self.mjd, timetype='UTC', ttype='DFT', noise=False, reorder=reorder, verbose=False)
                self.obslist.append(obs)

            uvlist = []
            antenna_list = []
            th_sigma = []
            site_pairs = []
            for obs in self.obslist:
                for tdata in obs.tlist():
                    num_antenna = len(np.unique(tdata['t1'])) + 1
                    if num_antenna < 3:
                        continue
                    antenna_list.append(num_antenna)
                    u = tdata['u']
                    v = tdata['v']

                    uvlist.append(np.stack((u, v), axis=-1)) 

                    sites = obsh.recarr_to_ndarr(tdata[['t1', 't2']], 'U32')
                    site_pairs.append(sites)

                    bw = obs.bw
                    tint = tdata['tint']
                    sig_rr = np.fromiter((obsh.blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'],
                                                        obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)
                                            for i in range(len(tint))), float)
                    sig_ll = np.fromiter((obsh.blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdl'],
                                                        obs.tarr[obs.tkey[sites[i][1]]]['sefdl'], tint[i], bw)
                                            for i in range(len(tint))), float)

                    sig_iv = 0.5*np.sqrt(sig_rr**2 + sig_ll**2)

                    th_sigma.append(sig_iv)
                    

            self.th_sigma = th_sigma
            self.site_pairs = site_pairs
            self.uvlist = uvlist
            self.antenna_list = antenna_list
            self.uvf = [0]
        else:
            self.uvf = np.load(filename)    
            self.antenna = 7
            self.atriads, self.btriads = self.Triads(self.antenna)

    def FTCI(self, imgs, add_th_noise=False, return_uv=False, method=1, th_noise_factor=1):
        imgs = torch.tensor(imgs).to(device)
        if self.ehtim:
            ci = np.array([np.array([]) for i in range(len(imgs))])
            out_uv = []
            metadata = []
            for uv, num_antenna, sig_iv, pairs in zip(self.uvlist, self.antenna_list, self.th_sigma, self.site_pairs):
                vis = self.Visibilities(imgs, torch.tensor(uv, dtype=torch.float32).to(device), self.fovx, self.fovy)
                if add_th_noise:
                    vis = vis + torch.tensor(obsh.cerror(sig_iv*th_noise_factor)).to(device)
                temp_ci, temp_uv = self.ClosureInvariants(vis, uv, num_antenna, method, pairs=pairs)
                temp_ci = temp_ci.reshape(imgs.shape[0], -1).cpu().detach().numpy()
                ci = np.concatenate((ci, temp_ci), axis=1)
                out_uv.append(temp_uv)
                metadata.append((num_antenna, len(temp_ci[0])))
            ci = torch.tensor(ci)
            if return_uv:
                out_uv = np.concatenate(out_uv, axis=-1)
                metadata = np.array(metadata)
                return ci, out_uv, metadata

        else:
            vis = self.Visibilities(imgs)
            ci = self.ClosureInvariants(vis)
        
        # reverse order of ci
        # ci = torch.flip(ci, [1])

        return ci
    

    def Visibilities(self, imgs, uv=None, xfov=225, yfov=225):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (torch.Tensor): tensor of images

        Returns:
            vis (torch.Tensor): visibilities taken for each image
        """
        if not self.ehtim:
            uv = torch.cat([self.uvf[x] for x in self.uvf])
        vis = self.DFT(imgs, uv, xfov, yfov)
        return vis.reshape((len(imgs), len(self.uvf), -1))


    def ClosureInvariants(self, vis, uv=None, n:int=7, method=1, pairs=None):
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
        if method == 1:
            if self.ehtim:
                unique_ant = pd.unique(pairs.flatten())
                ant_pairs = np.array([np.where(i == unique_ant)[0][0] for i in pairs.flatten()]).reshape(-1, 2)
                ant_pairs = [tuple(i) for i in ant_pairs]
                reverse_idx = [i for i in range(len(ant_pairs)) if ant_pairs[i][0] > ant_pairs[i][1]]
                ant_pairs = [ant_pairs[i] if i not in reverse_idx else (ant_pairs[i][1], ant_pairs[i][0]) for i in range(len(ant_pairs))]
                vis[:,:,reverse_idx] = torch.conj(vis[:,:,reverse_idx])
                self.atriads, self.btriads = self.Triads(n, pairs=ant_pairs)
            C_oa = vis[:, :, self.btriads[:, 0]]
            C_ab = vis[:, :, self.btriads[:, 1]]
            C_bo = torch.conj(vis[:, :, self.btriads[:, 2]])
            A_oab = C_oa / torch.conj(C_ab) * C_bo
            A_oab = torch.dstack((A_oab.real, A_oab.imag))
            A_max = nanmax(torch.abs(A_oab), dim=-1, keepdim=True)[0]
            ci = A_oab / A_max

            if uv is not None:
                uv = uv.T
                uv = uv.reshape(1,2, -1)
                uv0 = uv[:, :, self.btriads[:, 0]]
                uv1 = uv[:, :, self.btriads[:, 1]]
                uv2 = uv[:, :, self.btriads[:, 2]]
                uv = np.dstack((uv0,  uv1,  uv2))
                uv = uv.reshape(1, 2, 3, -1)
                return ci, uv
    
        elif method == 2:
            # Old Method, requires specific baseline arrangement
            # element_ids = np.arange(0, n, 1)
            # element_pairs = [(element_ids[i], element_ids[j]) for i in range(len(element_ids)) for j in range(i+1,len(element_ids))]

            element_pairs = pairs
            # _, idx = np.unique(element_pairs.reshape(-1), return_index=True)
            # element_ids = element_pairs.reshape(-1)[np.sort(idx)]
            element_ids = pd.unique(np.array(element_pairs).ravel())
            triads_indep = GU.generate_triangles(element_ids, baseid=element_ids[0])
            corrs_lol = SI.corrs_list_on_loops(vis.detach().cpu().numpy(), element_pairs, triads_indep, bl_axis=-1)
            advariants = SI.advariants_multiple_loops(corrs_lol)
            ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts='max', normpower=1)
            ci = torch.tensor(ci)
            if uv is not None: # need to fix triads_indep to be index and not element id
                uv = uv.T
                uv = uv.reshape(1,2, -1)
                triads_indep = np.array(triads_indep)
                triads_indep = np.array([[np.where(element_ids == i)[0] for i in j] for j in triads_indep])
                triads_indep = triads_indep.reshape(-1, 3)
                uv0 = uv[:, :, np.array(triads_indep)[:, 0]]
                uv1 = uv[:, :, np.array(triads_indep)[:, 1]]
                uv2 = uv[:, :, np.array(triads_indep)[:, 2]]
                uv = np.dstack((uv0,  uv1,  uv2))
                uv = uv.reshape(1, 2, 3, -1)
                return ci, uv
            
            return ci, None 
        
        return ci


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
        x = -2*np.pi*torch.matmul(uv,lm)[None, ...].to(device)
        visr = torch.sum(imgvect[:, None, :] * torch.cos(x).to(device), axis=-1)
        visi = torch.sum(imgvect[:, None, :] * torch.sin(x).to(device), axis=-1)
        if data.ndim == 2:
            vis = visr.ravel() + 1j*visi.ravel()
        else:
            vis = visr.ravel() + 1j*visi.ravel()
            vis = vis.reshape(out_shape)
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
