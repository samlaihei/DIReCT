from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder
from models.transformer import ViT
from models.recorder import Recorder



class DIReCT(nn.Module):
    def __init__(self, data_dim:int=992, latent_size:int=4, k_classes:int=5, imgdim1:int=64, imgdim2:int=64):
        super(DIReCT, self).__init__()

        self.latent_size = latent_size
        self.k_classes = k_classes
        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)
        self.ci_dim = data_dim

        # Encoder # input img, output n latent features
        self.encoder = Encoder(1, self.latent_size, n_res_layers=2, res_h_dim=8)

        # Decoder # input n, output img
        self.decoder = Decoder(self.latent_size, self.latent_size, n_res_layers=2, res_h_dim=8)

        # take ci and get latent features
        self.ci_xtrans = ViT(
            image_size=32, 
            # channels=1,
            patch_size=1, 
            num_classes=1024, 
            dim=64,
            depth=4, 
            heads=8, 
            mlp_dim=1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # self.attn = SelfAttentionLayer()

        # linear classifier
        self.classifier = nn.Sequential( # input is n, output is k one-hot classes
            nn.Linear(self.latent_size*16**2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.k_classes),
            nn.Softmax(dim=1)
        )

    def ci_latent(self, ci):
        # ci = ci.reshape(-1, 1, 1, self.ci_dim)
        ci = ci.reshape(-1, self.ci_dim)
        ci = ci.repeat(1, 3)
        ci = ci.reshape(-1, 3, self.ci_dim)
        ci = ci.reshape(-1, 3, 31, 32)
        features_ci = self.ci_xtrans(ci)
        features_ci = features_ci.reshape(-1, self.latent_size, self.imgdim1//4, self.imgdim2//4)

        return features_ci

    def ci_attn(self, ci):
        # ci = ci.reshape(-1, 1, 1, self.ci_dim)
        ci = ci.reshape(-1, self.ci_dim)
        ci = ci.repeat(1, 3)
        ci = ci.reshape(-1, 3, self.ci_dim)
        ci = ci.reshape(-1, 3, 31, 32)
        self.ci_xtrans = Recorder(self.ci_xtrans)
        features_ci, attns = self.ci_xtrans(ci)
        self.ci_xtrans = self.ci_xtrans.eject()
        features_ci = features_ci.reshape(-1, self.latent_size, self.imgdim1//4, self.imgdim2//4)
        return features_ci, attns
    
    def forward(self, imgs, ci, mask=None): 
        features_vae, features_q, recon_img = self.encoder_to_img(imgs)

        if mask is not None:
            ci = ci.masked_fill(mask==0, -1e9)

        features_ci, pred_img = self.predict(ci)

        features_cls = features_q.reshape(-1, features_q.shape[1]*features_q.shape[2]*features_q.shape[3])

        pred_class = self.classifier(features_cls)

        return features_vae, features_q, features_ci, recon_img, pred_img, pred_class
    
    def predict(self, ci): # get reconstructed image from ci
        features_pred = self.ci_latent(ci.reshape(-1, self.ci_dim, 1, 1))
        pred_img = self.decoder(features_pred)
        return features_pred, pred_img
    
    def predict_with_attn(self, ci):
        features_pred, attns = self.ci_attn(ci.reshape(-1, self.ci_dim, 1, 1))
        pred_img = self.decoder(features_pred)
        return features_pred, pred_img, attns
    
    def predict_class(self, ci):
        features_cls = self.ci_latent(ci.reshape(-1, self.ci_dim, 1, 1))
        features_cls = features_cls.reshape(-1, features_cls.shape[1]*features_cls.shape[2]*features_cls.shape[3])
        pred_class = self.classifier(features_cls)
        return pred_class
    
    def encoder_to_img(self, imgs):
        features_vae = self.encoder(imgs)
        features_q = features_vae
        recon_img = self.decoder(features_q)
        return features_vae, features_q, recon_img



if __name__ == "__main__":
    N = 32
    imgdim = 64
    latent_size = 4
    data_size = 992

    # random data
    x = np.random.random_sample((N, 1, imgdim, imgdim))
    x = torch.tensor(x).float()

    ci = np.random.random_sample((N, data_size))
    ci = torch.tensor(ci).float()
    mask = np.ones((N, data_size))
    mask = torch.tensor(mask).float()

    # test vae
    model = DIReCT(data_dim=data_size, latent_size=latent_size, k_classes=5)

    
    features_vae, features_q, features_ci, recon_img, pred_img, pred_class = model(x, ci, mask=mask)

    print('Features vae shape:', features_vae.shape)
    print('Features q shape:', features_q.shape)
    print('Features ci shape:', features_ci.shape)
    print('Recon img shape:', recon_img.shape)
    print('Pred img shape:', pred_img.shape)
    print('Pred class shape:', pred_class.shape)

