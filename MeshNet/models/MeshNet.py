import torch
import torch.nn as nn
from .layers import SpatialDescriptor, StructuralDescriptor, MeshConvolution


class MeshNet(nn.Module):

    def __init__(self, cfg, require_fea=False):
        super(MeshNet, self).__init__()
        self.require_fea = require_fea

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'], 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'], 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mask_ratio = cfg['mask_ratio']
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
            nn.Linear(256, cfg['num_classes'])
        )

    def forward(self, centers, corners, normals, neighbor_index):
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1)) # b, c, n
        if self.training:
            fea = fea[:, :, torch.randperm(fea.size(2))[:int(fea.size(2) * (1 - self.mask_ratio))]]
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1) # (n, 1024)
        return fea
        # fea = self.classifier[:-1](fea) # (n, 256)
        # cls = self.classifier[-1:](fea)

        # if self.require_fea:
        #     return cls, fea / torch.norm(fea)
        # else:
        #     return cls
