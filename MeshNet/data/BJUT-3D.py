import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler



class BJUT3D(Dataset):
    def __init__(self, data_dir, datafile, augment_data=True, jitter_sigma=0.01, jitter_clip=0.05):
        self.data_dir = data_dir
        self.datafile = datafile
        self.augment_data = augment_data
        if self.augment_data:
            self.jitter_sigma = jitter_sigma
            self.jitter_clip = jitter_clip

        self.data = []
        with open(self.datafile, 'r') as f:
            lines = f.readlines()
        for line in lines:
            linesplit = line.split('\n')[0].split()
            filename = linesplit[0]
            score = float(linesplit[1])
            self.data.append((os.path.join(self.data_dir, filename), score))

    def __getitem__(self, i):
        path, score = self.data[i]
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # data augmentation
        if self.augment_data and self.part == 'train':
            # jitter
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * self.jitter_clip, self.jitter_clip)
            face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(score, dtype=torch.float)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)