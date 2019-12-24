import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules import * 
from data.force_field import from_array_to_ff_batch


class ForceFiledCapsNet(nn.Module):
    def __init__(self, device_ids=None, grid_size=50, group_size=1, num_prim_caps=32, dim_prim_vect=4,
                 num_digit_caps=166, dim_digit_vect=8):
        super(ForceFiledCapsNet, self).__init__()

        self.array2ff = Lambda(from_array_to_ff_batch, grid_size=grid_size, group_size=group_size)
        self.array2ff = RaggedDataParallel(self.array2ff, device_ids=device_ids)

        self.conv_layer = ConvLayer(
            in_channels=2,
            out_channels=128,
            grid_size=grid_size,
        )
        self.conv_layer = nn.DataParallel(self.conv_layer, device_ids=device_ids)
        self.primary_capsules = PrimaryCaps(
            num_capsules=num_prim_caps,
            in_channels=128,
            out_channels=dim_prim_vect,
        )
        self.primary_capsules = nn.DataParallel(self.primary_capsules, device_ids=device_ids)
        self.digit_capsules = DigitCaps(
            num_capsules=num_digit_caps,
            num_routes=num_prim_caps * 3 ** 3,
            in_channels=dim_prim_vect,
            out_channels=dim_digit_vect,
        )
        self.digit_capsules = nn.DataParallel(self.digit_capsules, device_ids=device_ids)
        self.decoder = Decoder(
            in_capsules=num_digit_caps,
            in_channels=dim_digit_vect,
            grid_size=grid_size,
        )
        self.decoder = nn.DataParallel(self.decoder, device_ids=device_ids)

    def forward(self, array, fp):
        ff = self.array2ff(*array).squeeze()
        v = self.digit_capsules(
            self.primary_capsules(
                self.conv_layer(ff)
            )
        )
        fp = torch.tensor(fp, dtype=torch.float, device=v.device)
        # mask = fp.unsqueeze(-1)  # fp as mask
        mask = torch.ones(fp.unsqueeze(-1).shape, dtype=torch.float, device=v.device)  # mask removed
        reconstructions = self.decoder(v, mask)
        return v, fp, ff, reconstructions

    def infer(self, array):
        with torch.no_grad():
            self.eval()
            ff = self.array2ff(*array).squeeze()
            v = self.digit_capsules(
                self.primary_capsules(
                    self.conv_layer(ff)
                )
            )
        return v

    def loss(self, v, fp, ff, reconstructions):
        return (
                self.margin_loss(v, fp) +
                self.reconstruction_loss(ff, reconstructions)
        )

    @staticmethod
    def margin_loss(v, fp):
        # fp.shape: BS * bits
        v_c = torch.norm(v, p=2, dim=-1)
        left = F.relu(0.9 - v_c) ** 2
        right = F.relu(v_c - 0.1) ** 2

        loss = fp * left + 0.5 * (1.0 - fp) * right
        loss = loss.mean()
        return loss

    @staticmethod
    def reconstruction_loss(ff, reconstructions, weight=1):
        loss = weight * F.mse_loss(ff, reconstructions)
        return loss
