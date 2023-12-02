import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch

def seq_collate_irregular(data):
    (data, time,static,mask,gt) = zip(*data)

    data = torch.stack(data,dim=0)
    # data = torch.from_numpy(data)
    time = torch.stack(time,dim=0)
    static = torch.stack(static,dim=0)
    mask = torch.stack(mask,dim=0)
    gt = torch.stack(gt,dim=0)
    data = {
        'data': data,
        'time': time,
        'static':static,
        'mask':mask,
        'gt':gt
    }
    return data

def seq_collate_irregular_wo_static(data):
    (data, time,mask,gt) = zip(*data)

    data = torch.stack(data,dim=0)
    # data = torch.from_numpy(data)
    time = torch.stack(time,dim=0)
    # static = torch.stack(static,dim=0)
    mask = torch.stack(mask,dim=0)
    gt = torch.stack(gt,dim=0)
    data = {
        'data': data,
        'time': time,
        'mask':mask,
        'gt':gt
    }
    return data


class medical_dataloader(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, root, split_path, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(medical_dataloader, self).__init__()

        data_root = root

        split = np.load(split_path,allow_pickle=True)
        if training:
            index = split[0]
        else:
            index = split[2]
        self.data = np.load(data_root+'array.npy')[index,:]

        self.time = np.load(data_root+'time.npy')[index,:]
 
        self.gt = np.load(data_root+'gt.npy')[index,:]

        self.static = np.load(data_root+'static.npy')[index,:]

        self.mask = np.load(data_root+'mask.npy')[index,:]


        self.data = torch.from_numpy(self.data).type(torch.float)
        self.time = torch.from_numpy(self.time).type(torch.float)
        self.gt = torch.from_numpy(self.gt).type(torch.float)
        self.static = torch.from_numpy(self.static).type(torch.float)
        self.mask = torch.from_numpy(self.mask).type(torch.int)

        self.batch_len = len(self.data)
        print(self.batch_len)
        print(self.data.shape)
        print(self.static.shape)

        

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):

        data = self.data[index]
        time = self.time[index]
        gt = self.gt[index]
        static = self.static[index]
        mask = self.mask[index]##########
        
        out = \
            [data,time,static,mask,gt]
            
        return out


class medicalp12_dataloader(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, root, split_path, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(medicalp12_dataloader, self).__init__()

        data_root = root

        split = np.load(split_path,allow_pickle=True)
        if training:
            index = split[0]
        else:
            index = split[2]
        self.data = np.load(data_root+'array.npy')[index,:]

        self.time = np.load(data_root+'time.npy')[index,:]
 
        self.gt = np.load(data_root+'gt_los.npy')[index,:]

        self.static = np.load(data_root+'static.npy')[index,:]

        self.mask = np.load(data_root+'mask.npy')[index,:]


        self.data = torch.from_numpy(self.data).type(torch.float)
        self.time = torch.from_numpy(self.time).type(torch.float)
        self.gt = torch.from_numpy(self.gt).type(torch.float)
        self.static = torch.from_numpy(self.static).type(torch.float)
        self.mask = torch.from_numpy(self.mask).type(torch.int)

        self.batch_len = len(self.data)
        print(self.batch_len)

        

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):

        data = self.data[index]
        time = self.time[index]
        gt = self.gt[index]
        static = self.static[index]
        mask = self.mask[index]##########
        
        out = \
            [data,time,static,mask,gt]
            
        return out


class medicalpam_dataloader(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, root, split_path, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(medicalpam_dataloader, self).__init__()

        # data_root = '/home/ubuntu/PAMAP2data/trans_data/'
        data_root = root

        split = np.load(split_path,allow_pickle=True)
        if training:
            index = split[0]
        else:
            index = split[2]
        self.data = np.load(data_root+'array.npy')[index,:]

        val_data = self.data[self.data<1e9]
        mean = np.mean(val_data)
        std = np.std(val_data)
        self.data = (self.data-mean)/std
        self.time = np.load(data_root+'time.npy')[index,:]
 
        self.gt = np.load(data_root+'gt.npy')[index,:]

        # self.static = np.load(data_root+'static.npy')[index,:]

        self.mask = np.load(data_root+'mask.npy')[index,:]


        self.data = torch.from_numpy(self.data).type(torch.float)
        self.time = torch.from_numpy(self.time).type(torch.float)
        self.gt = torch.from_numpy(self.gt).type(torch.float)
        # self.static = torch.from_numpy(self.static).type(torch.float)
        self.mask = torch.from_numpy(self.mask).type(torch.int)

        self.batch_len = len(self.data)
        print(self.data.shape)

        

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):

        data = self.data[index]
        time = self.time[index]
        gt = self.gt[index]

        mask = self.mask[index]##########
        
        out = \
            [data,time,mask,gt]
            
        return out

