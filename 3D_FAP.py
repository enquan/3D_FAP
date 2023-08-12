import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
from MobileNetv2 import *
from fusion import *
import importlib
import argparse
from itertools import chain
from MeshNet.models.MeshNet import MeshNet
import yaml
from torch.utils.tensorboard import SummaryWriter
import time

def get_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return cfg

cfg = get_config('./MeshNet/config/BJUT3D_train.yaml')

parser = argparse.ArgumentParser()

# LDL
parser.add_argument('--single_label', action='store_true')
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--gamma', type=int, default=1)
parser.add_argument('--sample', type=str, default='L')
parser.add_argument('--loss1', type=str, default='ED')
parser.add_argument('--loss1_option',type=str, default='mean')
parser.add_argument('--loss3', type=str, default='3')
parser.add_argument('--loss3_option',type=str, default='sum')
parser.add_argument('--losses', type=str, default='123')
parser.add_argument('--interval', type=float, default=0.1)
parser.add_argument('--min_score', type=int, default=1)
parser.add_argument('--max_score', type=int, default=10)

# Hyperparameter
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--init_lr', action='store_true')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epoch', type=int, default=90)

# GPU
parser.add_argument('--device', type=str, default='0')

# 3D Data
parser.add_argument('--data_type', type=str, default='12')  # 1 - texture, 2 - pointcloud, 3 - mesh

# Point Cloud
parser.add_argument('--model', type=str, default='my_pointnet2')
parser.add_argument('--point', type=int, default=8192)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--sa1', type=int, default=1)

# Cyclical Learning Rates
parser.add_argument('--new_lr_scheme', action='store_true')
parser.add_argument('--stepsize', type=int)
parser.add_argument('--cycle', type=int, default=4)
parser.add_argument('--min_lr', type=float)
parser.add_argument('--max_lr', type=float)

global args
args = parser.parse_args()

# random seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# setup_seed(20)

# device configuration
device = torch.device("cuda:"+args.device) if torch.cuda.is_available() else 'cpu'

# hyper-parameter
if args.init_lr:
    num_epochs = 100
    learning_rate = 1e-6
if args.new_lr_scheme:
    num_epochs = args.stepsize * args.cycle * 2
    learning_rate = args.min_lr
if (not args.init_lr) and (not args.new_lr_scheme):
    num_epochs = args.epoch
    learning_rate = args.lr
batch_size = args.batch

def calc_lr_linear(epoch, min_lr=args.min_lr, max_lr=args.max_lr, stepsize=args.stepsize):
    k = (max_lr - min_lr) / (stepsize - 1)
    b1 = min_lr - k
    b2 = min_lr + 2 * stepsize * k
    epoch %= (stepsize * 2)
    if epoch == 0: 
        epoch = 1
    if epoch <= stepsize:
        curr_lr = k * epoch + b1
    else:
        curr_lr = -k * epoch + b2
    print("curr_lr="+str(curr_lr))
    return curr_lr

def calc_lr(epoch, min_lr=args.min_lr, max_lr=args.max_lr, stepsize=args.stepsize):
    exp = (max_lr / min_lr) ** (1 / int(stepsize - 1))
    epoch %= (stepsize * 2)
    if epoch == 0: 
        epoch = 1
    if epoch > stepsize:
        epoch = 2 * stepsize - epoch + 1
    curr_lr = min_lr * np.power(exp, epoch - 1) 
    print("curr_lr="+str(curr_lr))
    return curr_lr


# transform
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}

def cdf(x, mean, std):
    if args.sample == 'L':
        b = std / np.sqrt(2)
        return 0.5*(1+np.sign(x-mean)*(1-np.exp(-np.abs(x-mean)/b)))
    if args.sample == 'G':
        return norm.cdf(x, mean, std)

def sampling(x, mean, std):
    if args.sample == 'G':
        # std = 2
        return np.exp(-(np.power(x-mean,2))/(2*np.power(std,2)))/(std*np.sqrt(2*np.pi))
    elif args.sample == 'L':
        # b = np.sqrt(2)
        b = std / np.sqrt(2)
        return np.exp(-np.abs(x-mean)/b)/(2*b)

class BJUT3D(Dataset):
    def __init__(self, img_data, img_info, img_transform, point_data, num_point, mesh_data, max_faces, normalize_pc=True, augment_mesh_data=True, jitter_sigma=0.01, jitter_clip=0.05):
        '''
        img_data: 图片路径
        img_info: 图片信息
        img_transform: 图片变换
        point_data: 点云数据路径
        num_point: 用于训练的点数量
        mesh_data: 网格数据路径
        max_faces: 最大面数量
        augment_mesh_data: 是否增广网格数据
        jitter_sigma, jitter_clip: 网格数据增广参数
        '''
        self.img_data = img_data
        self.img_info = img_info
        self.img_transform = img_transform
        self.point_data = point_data
        self.num_point = num_point
        self.mesh_data = mesh_data
        self.max_faces = max_faces
        self.normalize_pc = normalize_pc
        self.augment_mesh_data = augment_mesh_data
        if self.augment_mesh_data:
            self.jitter_sigma = jitter_sigma
            self.jitter_clip = jitter_clip

        self.data_label_list = self.read_data(img_data=self.img_data, img_info=self.img_info, point_data=self.point_data, mesh_data=self.mesh_data)
        self.len = len(self.data_label_list)

    def __getitem__(self, index):
        # Image
        img_path, point_path, mesh_path, img_filename, attr, rating, score = self.data_label_list[index]
        img = Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        # Pointcloud
        with open(point_path, 'r') as f:
            temp = np.fromfile(f, np.int32)
            vertex_num, mesh_num = temp[0] - 10, temp[1] - 30
        with open(point_path, 'r') as f:
            temp = np.fromfile(f, np.float32)[2:]
            coo = temp[: vertex_num * 3].reshape(-1,3)
            rgb = temp[vertex_num * 3 : vertex_num * 3 * 2].reshape(-1,3)
            # mesh = data_array[vertex_num*3*2:].reshape(-1,3)
            point = np.concatenate((coo, rgb), axis=1).transpose(1,0)
            idx = np.random.choice(point.shape[1], self.num_point, replace=False)
            point = point[:,idx]
            if self.normalize_pc:
                point[0 : 3, :] = np.transpose(self.pc_normalize(np.transpose(point[0 : 3, :])))
        
        # Mesh
        mesh_raw_data = np.load(mesh_path)
        face = mesh_raw_data['faces']
        neighbor_index = mesh_raw_data['neighbors']
        # mesh data augmentation
        if self.augment_mesh_data:
            # jitter
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * self.jitter_clip, self.jitter_clip)
            face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)
        # random remove extra faces when n > max_faces
        num_face = len(face)
        if num_face > self.max_faces:
            new_face = []
            new_neighbor_index = []
            for i in range(self.max_faces):
                index = np.random.randint(0, num_face)
                new_face.append(face[index])
                new_neighbor_index.append(neighbor_index[index])
            # print(len(new_face))
            # print(len(new_neighbor_index))
            face = np.array(new_face)
            neighbor_index = np.array(new_neighbor_index)
        # fill for n < max_faces with randomly picked faces
        if num_face < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_face):
                index = np.random.randint(0, num_face)
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
        return img_filename, img, point, centers, corners, normals, neighbor_index, attr, rating, score

    def __len__(self):
        return self.len
    
    def read_data(self, img_data, img_info, point_data, mesh_data):
        output = []
        with open(img_info, 'r') as f:
            lines = f.readlines()
        for line in lines:
            linesplit = line.split('\n')[0].split()
            filename = linesplit[0]
            score = float(linesplit[1])
            std = float(linesplit[2])
            rating = [float(linesplit[l]) for l in range(3, 4 + args.max_score - args.min_score)]
            rating = torch.Tensor(rating)
            x = np.arange(args.min_score, args.max_score + args.interval, args.interval)
            attr = [cdf(x[i+1],score,std)-cdf(x[i],score,std) for i in range(x.shape[0] - 1)]
            attr = [j if j > 1e-15 else 1e-15 for j in attr]
            attr = torch.Tensor(attr)
            attr = torch.sigmoid(attr)
            attr = torch.nn.functional.normalize(attr, p=1, dim=0)
            img_path = os.path.join(img_data, filename)
            point_path = os.path.join(point_data, filename[:-4])
            mesh_path = os.path.join(mesh_data, filename[:-4]+'.npz')
            output.append((img_path, point_path, mesh_path, filename, attr, rating, score))
        return output
    
    def pc_normalize(self, pc):
        # l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


# loading the train/test data
img_data = './BJUT-3D_texture'
img_info_train = './train_full_texture.txt'
img_info_test = './test_full_texture.txt'
point_data = './BJUT-3D'
mesh_data = './BJUT-3D_npz_10000/'

train_data = BJUT3D(img_data=img_data, img_info=img_info_train, img_transform=transform['train'], point_data=point_data, num_point=args.point, mesh_data=mesh_data, max_faces=10000, augment_mesh_data=True)
test_data = BJUT3D(img_data=img_data, img_info=img_info_test, img_transform=transform['test'], point_data=point_data, num_point=args.point, mesh_data=mesh_data, max_faces=10000, augment_mesh_data=False)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch, num_workers=8, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch, num_workers=8, pin_memory=True, shuffle=False)

# net definition
img_net = mobilenet_v2(pretrained=True)
img_in_channel = img_net.classifier[1].in_features
img_net.classifier[1] = nn.Linear(img_in_channel, (int)((args.max_score-args.min_score)/args.interval))
img_net = img_net.to(device)

if args.model == "my_pointnet2":
    point_net = importlib.import_module(args.model).get_model(num_classes=1,dropout=args.dropout,sa1=args.sa1).to(device)
else:
    point_net = importlib.import_module(args.model).get_model(num_classes=1).to(device)

mesh_net = MeshNet(cfg=cfg['MeshNet'], require_fea=True).to(device)

fusion_in_channel = 0
if '1' in args.data_type:
    fusion_in_channel += 1280
if '2' in args.data_type:
    if args.model == "my_pointnet2":
        fusion_in_channel += 512
    else:
        fusion_in_channel += 1024
if '3' in args.data_type:
    fusion_in_channel += 1024
if args.single_label:
    fusion_net = fusion_net(in_channel=fusion_in_channel, out_channel=1, single_label=True).to(device)
else:
    fusion_net = fusion_net(in_channel=fusion_in_channel, out_channel=(int)((args.max_score-args.min_score)/args.interval)).to(device)

# loss
def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='batchmean')  # reduce=False
    outputs = torch.log(inputs+1e-15)  # sys.float_info.min
    loss = criterion(outputs, labels)
#     loss = loss.sum() / loss.shape[0]
    return loss

def L1_loss(inputs, labels):
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())
    return loss

def L1_dis(inputs, labels, args):
    loss = torch.sum(torch.abs(inputs-labels),dim=1)
    if args.loss1_option == 'mean':
        return loss.mean()
    elif args.loss1_option =='sum':
        return loss.sum()

def Euclidean_dis(inputs, labels, args, weight=None):
    if weight == None:
        loss = torch.pow(torch.sum(torch.pow((inputs-labels),2),dim=1),0.5)  # 开根号
    # loss = torch.sum(torch.pow((inputs-labels),2),dim=1)
    else:
        loss = torch.pow(torch.sum(weight*torch.pow((inputs-labels),2),dim=1),0.5)  # 开根号
    if args.loss1_option == 'mean':
        return loss.mean()
    elif args.loss1_option =='sum':
        return loss.sum()

L1 = nn.L1Loss()
L2 = nn.MSELoss()

def my_loss(outputs, target, args):
    if args.loss3 == '1':
        loss = torch.log(0.5*(torch.exp(outputs-target)+torch.exp(target-outputs)))
    elif args.loss3 == '2':
        loss = torch.log(1+torch.abs(outputs-target))
    elif args.loss3 == '3':
        loss = torch.exp(torch.abs(outputs-target))-1
    elif args.loss3 == '4':
        loss = torch.log(torch.abs(outputs-target)+torch.pow((1+torch.pow(outputs-target,2)),0.5))
    elif args.loss3 == '5':
        return 0
    if args.loss3_option == 'mean':
        return loss.mean()
    elif args.loss3_option == 'sum':
        return loss.sum()

# optimizer
optimizer = torch.optim.Adam(params=chain(img_net.parameters(),point_net.parameters(),mesh_net.parameters(),fusion_net.parameters()),lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)

# logger
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
if args.new_lr_scheme:
    log_dir = './runs/'+curr_time+'-'+'Data_type:'+args.data_type+'-'+str(num_epochs)+'-'+str(batch_size)+'-'+str(args.min_lr)+'-'+str(args.max_lr)+'/'
else:
    log_dir = './runs/'+curr_time+'-'+'Data_type:'+args.data_type+'-'+str(num_epochs)+'-'+str(batch_size)+'-'+str(learning_rate)+'/'
print(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# learning rate update
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# test
def test(test_data=True):
    img_net.eval()
    point_net.eval()
    mesh_net.train()
    fusion_net.eval()
    label = np.array([])
    pred = np.array([])
    if test_data:
        inference_loader = test_loader
    else:
        inference_loader = train_loader
    with torch.no_grad():
        for i, inputs in enumerate(inference_loader):
            img_filename, img, point, centers, corners, normals, neighbor_index, attr, rating, score = inputs # attr: (n, 90), score: (n)
            img_fea = None
            point_fea = None
            mesh_fea = None
            if '1' in args.data_type:
                img = img.to(device)
                img_fea = img_net(img)
            if '2' in args.data_type:
                point = point.to(device)
                point_fea = point_net(point)
            if '3' in args.data_type:
                centers = centers.to(device)
                corners = corners.to(device)
                normals = normals.to(device)
                neighbor_index = neighbor_index.to(device)
                mesh_fea = mesh_net(centers, corners, normals, neighbor_index)
            output = fusion_net(img_fea, point_fea, mesh_fea)
            if args.single_label:
                output = torch.squeeze(output)
                pred = np.append(pred, output.cpu().numpy())
            else:
                pred_score = torch.sum(output*rank, dim=1)  # (n)
                pred = np.append(pred, pred_score.cpu().numpy())
            label = np.append(label, score.cpu().numpy())
        print(label.shape)
        print(pred.shape)
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))
        if test_data:
            writer.add_scalar('Test/PC', correlation.item(), epoch + 1)
            writer.add_scalar('Test/MAE', mae.item(), epoch + 1)
            writer.add_scalar('Test/RMSE', rmse.item(), epoch + 1)
            writer.flush()
        else:
            writer.add_scalar('Train/PC', correlation.item(), epoch + 1)
            writer.add_scalar('Train/MAE', mae.item(), epoch + 1)
            writer.add_scalar('Train/RMSE', rmse.item(), epoch + 1)
            writer.flush()
    if test_data:
        print('Test Pearson Correlation: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(correlation, mae, rmse))
    else:
        print('Train Pearson Correlation: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(correlation, mae, rmse))


# train
def train(epoch, rank, alpha, beta, gamma):
    img_net.train()
    point_net.train()
    mesh_net.train()
    fusion_net.train()
    for i, inputs in enumerate(train_loader):
        img_filename, img, point, centers, corners, normals, neighbor_index, attr, rating, score = inputs # attr: (n, 90), score: (n)
        img_fea = None
        point_fea = None
        mesh_fea = None
        if '1' in args.data_type:
            img = img.to(device)
        if '2' in args.data_type:
            point = point.to(device)
        if '3' in args.data_type:
            centers = centers.to(device)
            corners = corners.to(device)
            normals = normals.to(device)
            neighbor_index = neighbor_index.to(device)
        if not args.single_label:
            attr = attr.to(device)
            rating = rating.to(device)
        score = score.to(device)

        optimizer.zero_grad()
        if '1' in args.data_type:
            img_fea = img_net(img)
            # print(img_fea.shape)
        if '2' in args.data_type:
            point_fea = point_net(point)
            # print(point_fea.shape)
        if '3' in args.data_type:
            mesh_fea = mesh_net(centers, corners, normals, neighbor_index)
            # print(mesh_fea.shape)
        outputs = fusion_net(img_fea, point_fea, mesh_fea)

        if not args.single_label:
            plevel = torch.zeros(outputs.shape[0], args.max_score - args.min_score + 1, dtype=torch.float).to(device)
            cnt = torch.tensor([10] * (args.max_score - args.min_score + 1)).to(device)
            cnt[0] -= 5
            cnt[-1] -= 5
            idx = [i.repeat(times) for i, times in zip(torch.arange(len(cnt)),cnt)]
            idx = torch.cat(idx).to(device)
            plevel.index_add_(dim=1, index=idx, source=outputs)
            scores = torch.sum(outputs*rank, dim=1)  # (n)
            if args.loss1 == 'L1':
                loss1 = L1_dis(outputs, attr, args)
            elif args.loss1 == 'ED':
                loss1 = Euclidean_dis(outputs, attr, args)
            elif args.loss1 == 'KL':
                loss1 = kl_loss(outputs, attr)
            loss2 = Euclidean_dis(plevel, rating, args, weight=None)
            if args.loss3 == 'L1':
                loss3 = L1(scores, score)
            else:
                loss3 = my_loss(scores, score, args)
            total_loss = 0
            if '1' in args.losses:
                total_loss += alpha * loss1
            if '2' in args.losses:
                total_loss += beta * loss2
            if '3' in args.losses:
                total_loss += gamma * loss3
        else:
            outputs = torch.squeeze(outputs)
            total_loss = L2(outputs, score.float())
        total_loss.backward()
        optimizer.step()

        if args.single_label:
            if (i + 1) == total_step:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, total_loss), flush=True)
                writer.add_scalar('Train/Loss', total_loss.item(), epoch + 1)
                writer.flush()
        else:
            if (i + 1) == total_step:
                print("Epoch [{}/{}], Step [{}/{}], Loss 1: {:.4f}, Loss 2: {:.4f}, Loss 3: {:.4f}, Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss1, loss2, loss3, total_loss), flush=True)
                writer.add_scalar('Train/Loss', loss1.item(), epoch + 1)
                writer.add_scalar('Train/Loss', loss2.item(), epoch + 1)
                writer.add_scalar('Train/Loss', loss3.item(), epoch + 1)
                writer.add_scalar('Train/Loss', total_loss.item(), epoch + 1)
                writer.flush()
    return total_loss.item()


total_step = len(train_loader)
curr_lr = learning_rate
rank = torch.Tensor([i for i in np.arange(args.min_score + 0.5 * args.interval, args.max_score + 0.5 * args.interval, args.interval)]).to(device)

for epoch in range(num_epochs):
    if args.new_lr_scheme:
        update_lr(optimizer, calc_lr(epoch + 1))
    _ = train(epoch=epoch, rank=rank, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    test(test_data=False) # training set
    test(test_data=True) # test set
    if args.init_lr:
        min_lr = 1e-6
        max_lr = 0.1
        curr_lr *= ((max_lr / min_lr) ** (1 / 100))
        print('curr_lr = ' + str(curr_lr))
        update_lr(optimizer, curr_lr)
    if (not args.init_lr) and (not args.new_lr_scheme) and ((epoch + 1) % 5 == 0):
        curr_lr /= 10
        update_lr(optimizer, curr_lr)
        print('curr_lr = ' + str(curr_lr))

# test()
