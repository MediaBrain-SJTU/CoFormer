import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from data.dataloader import medical_dataloader,medicalp12_dataloader,medicalpam_dataloader,seq_collate_irregular_wo_static,seq_collate_irregular
from models.model_medical_attn_aggre import make_model

from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score,accuracy_score
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, losses_str, log):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), losses_str), log)


def test():
    model.eval()
    all_loss = 0
    all_logits = []
    all_gt = []
    length = 0
    i = 0
    for data in tqdm(test_loader):
        # data = test_loader[22]
        array = data['data'].to(device)
        time = data['time'].to(device)
        static = data['static'].to(device)
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)

        prediction = model(array,time,mask,static)
        # print(i,torch.isnan(prediction).any())
        i += 1
        batch_num = prediction.shape[0]
        logits = nn.functional.softmax(prediction,dim=-1)
        all_logits.append(logits)
        all_gt.append(gt)

        gt = torch.nn.functional.one_hot(gt.to(torch.int64),2).squeeze(1)
        
        loss = criterion(prediction,gt.float())
        """ optimize """
        all_loss += loss
        length += batch_num

    all_logits = torch.cat(all_logits,dim=0)
    all_gt = torch.cat(all_gt,dim=0)

    auc = roc_auc_score(all_gt.cpu(), all_logits[:, 1].cpu())
    aupr = average_precision_score(all_gt.cpu(), all_logits[:, 1].cpu())
    print('auc',auc,'aupr',aupr,'loss',all_loss/length)
    return all_loss/length

def test_multiclass():
    model.eval()
    all_loss = 0
    all_logits = []
    all_gt = []
    length = 0
    for data in tqdm(test_loader):
        # data = test_loader[22]
        array = data['data'].to(device)
        time = data['time'].to(device)
        # static = data['static'].to(device)
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)

        prediction = model(array,time,mask)
        # print(i,torch.isnan(prediction).any())
        batch_num = prediction.shape[0]
        logits = nn.functional.softmax(prediction,dim=-1)
        all_logits.append(logits)
        all_gt.append(gt)
        gt = gt.squeeze(1)
        # gt = torch.nn.functional.one_hot(gt.to(torch.int64),8).squeeze(1)
        
        loss = criterion(prediction,gt.long())
        """ optimize """
        all_loss += loss
        length += batch_num
        


    all_logits = torch.cat(all_logits,dim=0)
    all_gt = torch.cat(all_gt,dim=0)
    
    all_pred = np.argmax(np.array(all_logits.cpu()),axis=1)

    # acc = np.sum(all_pred.ravel() == all_gt.ravel()) / all_pred.shape[0] 
    acc = accuracy_score(all_gt.cpu(), all_pred )
    precision = precision_score(all_gt.cpu(), all_pred, average='macro', )
    recall = recall_score(all_gt.cpu(), all_pred, average='macro', )
    F1 = f1_score(all_gt.cpu(), all_pred, average='macro', )
    print('acc',acc,'precision',precision,'recall',recall,'F1',F1)
    return all_loss/length



def train(epoch):

    model.train()

    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        array = data['data'].to(device)
        time = data['time'].to(device)
        static = data['static'].to(device)#####
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)
        # gt = gt.squeeze(1)
        prediction = model(array,time,mask,static)
        gt = torch.nn.functional.one_hot(gt.to(torch.int64),2).squeeze(1)#####
        loss = criterion(prediction,gt.float())
        """ optimize """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % cfg.print_freq == 0 :

            print('epoch',epoch,'iter',iter_num,'/',total_iter_num,'loss:',loss.item())
            
        iter_num += 1

    scheduler.step()

def train_multiclass(epoch):

    model.train()

    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        array = data['data'].to(device)
        time = data['time'].to(device)
        # static = data['static'].to(device)#####
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)
        gt = gt.squeeze(1)
        prediction = model(array,time,mask)
        gt = torch.nn.functional.one_hot(gt.to(torch.int64),8).squeeze(1)#####
        loss = criterion(prediction,gt.long())
        """ optimize """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % cfg.print_freq == 0 :

            print('epoch',epoch,'iter',iter_num,'/',total_iter_num,'loss:',loss.item())
            
        iter_num += 1

    scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='trans_medical')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=2)

    parser.add_argument('--input_dim',type=int,default=1)
    parser.add_argument('--output_dim',type=int,default=2)
    parser.add_argument('--num_layers',type=int,default=8)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--d_model',type=int,default=256)
    parser.add_argument('--d_ff',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--num_agents',type=int,default=34)
    parser.add_argument('--num_neighbors',type=int,default=30)
    parser.add_argument('--agent_encoding_dim',type=int,default=32)
    parser.add_argument('--static_dim',type=int,default=6)

    parser.add_argument('--split_path') # specify the split index
    parser.add_argument('--data_root') # specify the dataset root
    parser.add_argument('--dataset') # specify the dataset type ('p19','p12','pam')

    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
   

    device=torch.device("cuda",args.gpu)
    print('device',device)


    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')

    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)

    print('data prepare')
    """ data """
    if args.dataset == 'p19':
        train_dset = medical_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medical_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)
    
    elif args.dataset == 'p12':
        train_dset = medicalp12_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medicalp12_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)

    elif args.dataset == 'pan':
        train_dset = medicalpam_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medicalpam_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)
    
    
    
    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        num_workers=4,
        collate_fn=seq_collate_irregular,
        pin_memory=True)

    test_loader = DataLoader(
        test_dset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        collate_fn=seq_collate_irregular)

    print('model prepare')
    """ model """
    if args.dataset == 'pam':
        from models.model_medical_wo_static import make_model
    model = make_model(src_vocab=args.input_dim,tgt_vocab=args.output_dim,N=args.num_layers, d_model=args.d_model, d_ff=args.d_ff, h=args.heads, dropout=args.dropout,
                       num_agents=args.num_agents,num_neighbors=args.num_neighbors,agent_encoding_dim=args.agent_encoding_dim)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])

    """ start training """
    model.to(device)
    
    model.train()
    best_loss = 100
    for epoch in range(args.start_epoch, cfg.num_epochs):

        with torch.no_grad():
            if args.dataset == 'pam':
                current_loss = test_multiclass()
            else:
                current_loss = test()
        if current_loss<best_loss:
            best_loss = current_loss
        print('best loss:',best_loss)
        if args.dataset == 'pam':
            train_multiclass(epoch)
        else:
            train(epoch)
        

        print('start new epoch')
        # """ save model """
        if (cfg.model_save_freq > 0 and (epoch + 1) % cfg.model_save_freq == 0) or epoch == 0:
            cp_path = cfg.model_path % (epoch + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': epoch + 1}
            torch.save(model_cp, cp_path)








