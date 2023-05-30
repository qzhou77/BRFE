# USAGE EXAPMLES:
# 
# python GCNII.py --dataset cora --save_teacher
# python GCNII.py --dataset citeseer --layer 32 --hidden_channels 256 --theta 0.6 --dropout 0.7 --save_teacher
# python GCNII.py --dataset pubmed --layer 16 --hidden_channels 256 --theta 0.4 --dropout 0.5 --conv_decay 5e-4 --save_teacher

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN2Conv
import matplotlib.pyplot as plt
import pickle
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from termcolor import colored
import argparse
import time
import os
t = 0
tt = 0

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--conv_decay', type=float, default=0.01, help=' ')
parser.add_argument('--lin_decay', type=float, default=5e-4, help=' ')
parser.add_argument('--split', type=str, default="public", help=' ')
parser.add_argument('--save_teacher', action="store_true", help=' ')
parser.add_argument('--log_ep', type=int, default=2000, help=' ')
parser.add_argument('--epochs', type=int, default=1000, help=' ')
parser.add_argument('--hidden_channels', type=int, default=64, help=' ')
parser.add_argument('--layers', type=int, default=64, help=' ')
parser.add_argument('--dataset', type=str, default='cora', help=' ')
parser.add_argument('--theta', type=float, default=0.5, help=' ')
parser.add_argument('--alpha', type=float, default=0.1, help=' ')

args = parser.parse_args()


class GCNII(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, num_features=0, num_classes=0,
                norm = False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(num_features, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, num_classes))
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
        self.dropout = dropout
        self.norm = norm
    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        x_out_list = []
        for i,conv in enumerate(self.convs):
            x_out_list.append(x.detach())#-3...
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()
            if self.norm:
#                 x_u = x.mean()
#                 x_sigma = x.std()
#                 x = (x-x_u)/x_sigma
                rowsum = torch.div(1.0, torch.sum(x, dim=1))
                rowsum[torch.isinf(rowsum)] = 0.
                x = torch.mm(torch.diag(rowsum), x)
        x_out_list.append(x.detach())#-2
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        distr = x.log_softmax(dim=-1)
        x_out_list.append(distr.detach())#-1
        return distr,x_out_list
nll = torch.nn.NLLLoss()
kl = nn.KLDivLoss(reduction="sum",log_target=True)


dataset = args.dataset
dspath = "./data/"+dataset
split_type = ["public","full","geom-gcn","random"]
if args.split == 'public':
    i_s = 0
elif args.split == 'random':
    i_s = 3
elif args.split == 'full':
    i_s = 1
print(split_type[i_s])
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset_class = Planetoid(dspath, dataset, split=split_type[i_s], 
                          transform=transform)
data = dataset_class[0]

print(data)
print("train,val,test: ",data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
print("calss_num: ",dataset_class.num_classes)
data = data.cuda()
# data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
#                           sparse_sizes=(data.x.shape[0], data.x.shape[0])).t()
data.adj_t = gcn_norm(data.adj_t)

all_train = []
all_val = []
all_test = []
print("-------------------")
for run_n in range(args.runs):

    gcnii_model = GCNII(hidden_channels=args.hidden_channels, num_layers=args.layers, alpha=args.alpha, theta=args.theta, 
                        shared_weights=True, dropout=args.dropout,
                        num_features=data.x.shape[1],num_classes=dataset_class.num_classes,
                    norm=False).cuda()

    ##---------------##
    rebuttal_time=[]
    gcnii_model.eval()

    gcnii_optimizer = torch.optim.Adam([
        dict(params=gcnii_model.convs.parameters(), weight_decay=args.conv_decay, lr=0.01),
        dict(params=gcnii_model.lins.parameters(), weight_decay=args.lin_decay, lr=0.01)
    ])
    total_ep = args.epochs
    log_ep = args.log_ep
    eval_save = args.save_teacher
    best_val = 0
    best_ep = -1
    best_eval = True
    print_train = False

    gcnii_train_list = []
    gcnii_train_loss_list = []
    gcnii_val_list = []
    gcnii_test_list = []
    for ep in range(total_ep):
        gcnii_model.train()
        ep=ep+1
        gcnii_optimizer.zero_grad()
        gcnii_distr, gcnii_x_out_list= gcnii_model(data.x,data.adj_t)
        loss = nll(gcnii_distr[data.train_mask], data.y[data.train_mask])
        loss.backward()
        gcnii_train_loss_list.append(loss.item())
        gcnii_optimizer.step()
        with torch.no_grad():
            gcnii_model.eval()
            t1 = time.time()
            gcnii_distr, gcnii_x_out_list= gcnii_model(data.x,data.adj_t)
            t2 = time.time()
            t += t2-t1
            tt += 1

            pred = gcnii_distr.max(1)[1]
            right_pred = pred.eq(data.y)
            mask = data.train_mask
            gcnii_train_list.append(right_pred[mask].sum().item() / mask.sum().item())
            mask = data.val_mask
            gcnii_val_list.append(right_pred[mask].sum().item() / mask.sum().item())
            mask = data.test_mask
            gcnii_test_list.append(right_pred[mask].sum().item() / mask.sum().item())
        if gcnii_val_list[-1]>best_val:
            best_val = gcnii_val_list[-1]
            best_ep = ep
            torch.save(gcnii_model.state_dict(),'gcnii')
        if ep%log_ep==0:
            print(f'Epoch: {ep:03d}, Train: {gcnii_train_list[-1]:.4f}, \
                  Val: {gcnii_val_list[-1]:.4f}, Test: {gcnii_test_list[-1]:.4f}')

    if print_train:
        plt.plot(gcnii_train_list,c='b',alpha=0.3)
        plt.plot(gcnii_val_list,c='r',alpha=0.3)
        plt.plot(gcnii_test_list,c='g',alpha=0.3)
    if best_eval:

        gcnii_model.load_state_dict(torch.load("gcnii"))
        gcnii_model.eval()
        gcnii_distr, gcnii_x_out_list= gcnii_model(data.x,data.adj_t)

        pred = gcnii_distr.max(1)[1]
        right_pred = pred.eq(data.y)
        mask = data.train_mask
        gcnii_train_list.append(right_pred[mask].sum().item() / mask.sum().item())
        mask = data.val_mask
        gcnii_val_list.append(right_pred[mask].sum().item() / mask.sum().item())
        mask = data.test_mask
        gcnii_test_list.append(right_pred[mask].sum().item() / mask.sum().item())
        all_train.append(gcnii_train_list[-1])
        all_val.append(gcnii_val_list[-1])
        all_test.append(gcnii_test_list[-1])
        print(f'Best ep:{best_ep}:{best_val},Train: {gcnii_train_list[-1]:.4f}, \
            Val: {gcnii_val_list[-1]:.4f}, Test: {gcnii_test_list[-1]:.4f}')
    if eval_save:
        if not os.path.exists(f"./saved_teacher/{args.dataset}"):
            os.makedirs(f"./saved_teacher/{args.dataset}")
        gcnii_d_names = [f'./saved_teacher/{args.dataset}/{args.dataset}-gcnii_distr_run-{run_n}.d',f'./saved_teacher/{args.dataset}/{args.dataset}-gcnii_x_out_list_run-{run_n}.d']
        with open(gcnii_d_names[0],'wb') as f:
            pickle.dump(gcnii_distr,f)
        with open(gcnii_d_names[1],'wb') as f:
            pickle.dump(gcnii_x_out_list,f)
        torch.save(gcnii_model,f"./saved_teacher/{args.dataset}/{args.dataset}-gcnii_model_run-{run_n}_test-{gcnii_test_list[-1]:.4f}.d")
        print(colored("!!!saved!!!",'red'))
    else:
        print(colored("!!!not saved!!!",'blue'))


print("-------------------")
print("train:",f"{np.mean(all_train):.4f}, {np.std(all_train):.4f}",
        "; val:",f"{np.mean(all_val):.4f}, {np.std(all_val):.4f}", 
        "; test:",f"{np.mean(all_test):.4f}, {np.std(all_test):.4f}")