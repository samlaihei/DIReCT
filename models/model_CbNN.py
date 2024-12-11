from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import models.siren as siren
import torch.nn.functional as F

class CbNN():
    def __init__(self, clObj, cleanbeam, ci_target, img_target, stokes=False,
                  device=torch.device("cpu"), imgdim1:int=64, imgdim2:int=64):
        self.model = CbNNmodel(imgdim1=imgdim1, imgdim2=imgdim2).to(device)
        if stokes:
            siren_dim_out = 4
        else:
            siren_dim_out = 1

        self.siren = siren.SirenNet(
                        dim_in = 2,                        # input dimension, ex. 2d coor
                        dim_hidden = 256,                  # hidden dimension
                        dim_out = siren_dim_out,                       # output dimension, ex. rgb value
                        num_layers = 5,                    # number of layers
                        final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
                        w0_initial = 2.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
                    ).to(device)
        self.wrapper = siren.SirenWrapper(self.siren, imgdim1, imgdim2).to(device)

        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.nnloss = nn.L1Loss()
        self.clObj = clObj
        self.cleanbeam = cleanbeam
        self.ci_target = ci_target
        self.img_target = img_target
        self.stokes = stokes
        self.device = device

        coords = np.meshgrid(np.linspace(0, clObj.fovx, imgdim1), np.linspace(0, clObj.fovy, imgdim2))
        coords = np.stack(coords)
        coords = torch.tensor(coords).float().to(device)
        coords = coords.flatten(1,2).T
        self.coords = coords

        coords_flat = coords.flatten().unsqueeze(0)
        self.coords_flat = coords_flat


    def train(self, siren=False, nepochs:int=500, condition_epochs:int=100, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        if siren:
            self.trainSiren(nepochs=nepochs, condition_epochs=condition_epochs, ci_weight=ci_weight, init_lr=init_lr, lr_scale=lr_scale, verbose=verbose)
        else:
            self.trainCbNN(nepochs=nepochs, condition_epochs=condition_epochs, ci_weight=ci_weight, init_lr=init_lr, lr_scale=lr_scale, verbose=verbose)

    def trainCbNN(self, nepochs:int=500, condition_epochs:int=100, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr)
        criterion = self.loss_function
        l1 = lambda epoch: lr_scale ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l1)
        train_loss = []
        for epoch in tqdm(range(nepochs)):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.coords_flat, stokes=self.stokes)
            loss = criterion(outputs, self.ci_target, self.img_target, epoch=epoch, condition_epochs=condition_epochs, ci_weight=ci_weight)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

            if verbose and ((epoch+1) % 100 == 0 or epoch == 0):
                outputs = self.evaluate(plot=True)
                print(f'Epoch {epoch+1}, lr: {scheduler.get_last_lr()[0]}')
            scheduler.step()

    def trainSiren(self, nepochs:int=1000, condition_epochs:int=500, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        optimizer = torch.optim.AdamW(self.siren.parameters(), lr=init_lr)
        criterion = self.loss_function
        l1 = lambda epoch: lr_scale ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l1)
        train_loss = []
        for epoch in tqdm(range(nepochs)):
            self.siren.train()
            optimizer.zero_grad()
            outputs = self.siren(self.coords)
            outputs = outputs.reshape(1, self.siren.dim_out, self.imgdim1, self.imgdim2)
            loss = criterion(outputs, self.ci_target, self.img_target, epoch=epoch, condition_epochs=condition_epochs, ci_weight=ci_weight)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
            
            if verbose and ((epoch+1) % 100 == 0 or epoch == 0):
                outputs = self.evaluate(siren=True, plot=True)
                print(f'Epoch {epoch+1}, lr: {scheduler.get_last_lr()[0]}')
            scheduler.step()



    def loss_function(self, outputs, ci_target, img_target, epoch=0, condition_epochs:int=100, ci_weight=5):
        outputs = self.preprocess(outputs).to(torch.float64)
        outputs_ci = self.clObj.FTCI(outputs, add_th_noise=False, stokes=self.stokes)

        img_loss = self.nnloss(outputs, img_target)
        ci_loss = self.nnloss(outputs_ci, ci_target)

        # ci_loss = nn.SmoothL1Loss(beta=1e-3)(outputs_ci, ci_target)
        # ci_loss = nn.L1Loss()(outputs_ci, ci_target)
        # ci_loss = nn.MSELoss()(outputs_ci, ci_target)

        # output_mask = outputs[0][0] > 0

        if epoch < condition_epochs:
            return img_loss/img_loss.detach()
        else:
            return ci_loss/ci_loss.detach() * ci_weight + img_loss/img_loss.detach() #+ self.sl1(output_mask)

    def sl1(self, imvec):
        """L1 norm regularizer
        """
        l1 = torch.sum(torch.abs(imvec))
        return l1

    def preprocess(self, img):
        if self.stokes:
            out_img = torch.zeros_like(img)
            img4 = img[0]
            for ind, img in enumerate(img4):
                norm = torch.max(img)
                img = img/norm
                img = torch.fft.ifft2(torch.fft.fft2(img) * torch.fft.fft2(torch.tensor(self.cleanbeam.imvec.reshape(self.imgdim1, self.imgdim2), dtype=torch.float64).to(self.device)))
                img = torch.fft.fftshift(img)
                img = img.reshape(1, self.imgdim1, self.imgdim2)
                img = img.to(torch.float64)
                if ind == 0:
                    img = F.relu(img)
                    img = F.threshold(img, 1e-4, 0)
                out_img[:, ind, :, :] = img*norm
            return out_img
        else:
            norm = torch.max(img[0])
            img = img/norm
            img = F.threshold(img, 1e-4, 0)
            img = torch.fft.ifft2(torch.fft.fft2(img[0]) * torch.fft.fft2(torch.tensor(self.cleanbeam.imvec.reshape(self.imgdim1, self.imgdim2), dtype=torch.float64).to(self.device)))
            img = torch.fft.fftshift(img)
            img = img.reshape(1, 1, self.imgdim1, self.imgdim2)
            img = img.to(torch.float64) * norm
            return img
        
    def evaluate(self, siren=False, coords=None, plot=False):
        if siren:
            self.siren.eval()
            if coords is not None:
                outputs = self.siren(coords)
            else:
                outputs = self.siren(self.coords)
                outputs = outputs.reshape(1, self.siren.dim_out, self.imgdim1, self.imgdim2)
                outputs = self.preprocess(outputs)
        else:
            self.model.eval()
            outputs = self.model(self.coords_flat, stokes=self.stokes)
            outputs = self.preprocess(outputs)

        if plot:
            if self.stokes:
                _, axs = plt.subplots(1, 4, figsize=(4,1))
                plt.subplots_adjust(wspace=0, hspace=0)
                for i in range(4):
                    axs[i].imshow(outputs[0, i].detach().cpu().numpy(), cmap='afmhot')
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                plt.show()
            else:
                _, ax = plt.subplots(1, 1, figsize=(1,1))
                ax.imshow(outputs[0, 0].detach().cpu().numpy(), cmap='afmhot')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()

        return outputs

    def _replaceInitImg(self, siren=False):
        if siren:
            self.img_target = self.evaluate(siren=True).detach()
        else:
            self.img_target = self.evaluate().detach()
        return self.img_target

class CbNNmodel(nn.Module):
    def __init__(self, imgdim1:int=64, imgdim2:int=64):
        super(CbNNmodel, self).__init__()

        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.imgdim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.imgdim),
            nn.ReLU()
        )

        self.mlp_stokes = nn.Sequential(
            nn.Linear(2*self.imgdim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4*self.imgdim),
            # nn.ReLU()
        )

        self.stokes_layers = nn.ModuleList([])
        for i in range(3):
            self.stokes_layers.append(nn.Sequential(
                nn.Linear(2*self.imgdim, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.imgdim),
            ))
    
    def forward(self, coords, stokes=False): 
        if stokes:
            result = []
            # for i in range(4):
            #     if i == 0:
            #         result.append(self.mlp(coords))
            #     else:
            #         result.append(self.stokes_layers[i-1](coords))
            # result = torch.stack(result)
            # result = torch.swapaxes(result, 0, 1).reshape(-1, 4, self.imgdim1, self.imgdim2)

            # using the mlp_stokes architecture
            result = self.mlp_stokes(coords)
            result = result.reshape(-1, 4, self.imgdim1, self.imgdim2)
        else:
            result = self.mlp(coords)
            result = result.reshape(-1, 1, self.imgdim1, self.imgdim2)

        return result
    

