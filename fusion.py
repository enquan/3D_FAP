import torch
import torch.nn as nn
import torch.nn.functional as functional

class fusion_net(nn.Module):
    def __init__(self, in_channel, out_channel, single_label=False):
        super(fusion_net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.single_label = single_label
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.in_channel, self.out_channel),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, img_fea, point_fea, mesh_fea):
        concat_fea = [img_fea, point_fea, mesh_fea]
        while None in concat_fea:
            concat_fea.remove(None)
        fea = torch.cat(concat_fea, 1)
        x = self.classifier(fea)
        if not self.single_label:
            x = torch.sigmoid(x)
            x = functional.normalize(x, p=1, dim=1)
        return x