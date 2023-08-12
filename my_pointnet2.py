import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_classes,dropout,normal_channel=True,sa1=1):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.sa1 = sa1
        self.normal_channel = normal_channel
        if self.sa1 == 1:
            self.sa11 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 64, 128]]) # 320+3
            temp = 320
        elif self.sa1 == 2:
            self.sa12 = PointNetSetAbstractionMsg(512, [0.2, 0.4], [32, 128], in_channel,[[32, 32, 64], [64, 64, 128]]) # 192+3
            temp = 192
        elif self.sa1 == 3:
            # self.sa13 = PointNetSetAbstractionMsg(512, [0.4], [128], in_channel,[[64, 64, 128]]) # 128+3
            self.sa13 = PointNetSetAbstractionMsg(128, [0.4], [32], in_channel,[[16, 16, 32]]) # 32+3
            temp = 32
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, temp + 3, [128, 256, 512], True)
        # self.sa3 = PointNetSetAbstraction(None, None, None, temp + 3, [32, 64, 128], True)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, num_classes)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(256, num_classes)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        if self.sa1 == 1:
            l1_xyz, l1_points = self.sa11(xyz, norm)
        elif self.sa1 == 2:
            l1_xyz, l1_points = self.sa12(xyz, norm)
        elif self.sa1 == 3:
            l1_xyz, l1_points = self.sa13(xyz, norm)
        l3_xyz, l3_points = self.sa3(l1_xyz, l1_points)
        x = l3_points
        x = l3_points.view(B, 512)

        # x = self.bn1(x)
        # x = self.drop1(x)
        # x = torch.sigmoid(x)
        # x = self.fc1(x)
        # x = self.drop2(x)
        # x = self.fc2(x)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)


        return x


