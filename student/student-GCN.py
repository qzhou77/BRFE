# USAGE EXAPMLES:
# 
# python student-GCN.py
# python student-GCN.py --teacher_name 9
# 
# python student-GCN.py --hidden_channel 256 --dataset citeseer
# python student-GCN.py --hidden_channel 256 --dataset citeseer --teacher_name 9
# 
# python student-GCN.py --hidden_channel 256 --dataset pubmed
# python student-GCN.py --hidden_channel 256 --dataset pubmed --teacher_name 9
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import pickle
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import argparse
import time
t = 0
tt = 0


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--split', type=str, default="public", help=' ')
parser.add_argument('--layer', type=int, default=2, help=' ')
parser.add_argument('--hidden_channel', type=int, default=128, help=' ')
parser.add_argument('--lin', action='store_true', help=' ')
parser.add_argument('--save_teacher', action="store_true", help=' ')
parser.add_argument('--log_ep', type=int, default=2000, help=' ')
parser.add_argument('--epochs', type=int, default=1000, help=' ')
parser.add_argument('--teacher_name', type=str, default=None, help=' ')
parser.add_argument('--temp', type=float, default=1, help=' ')
parser.add_argument('--distill_range', type=str, default="train", help=' ')
parser.add_argument('--dataset', type=str, default='cora', help=' ')
parser.add_argument('--weight_decay', type=float, default=5e-4, help=' ')
parser.add_argument('--tmp_model_name', type=str, default='gcn', help=' ')
args = parser.parse_args()



class GCN(nn.Module):
    def __init__(self,hidden_channels, num_layers,
                dropout=0.0, num_features=0, num_classes=0,
                norm = False,lin=False):
        super().__init__()
        
        self.lin = lin
        if self.lin or True:
            self.lins = nn.Linear(hidden_channels, num_classes)
        
        self.convs = nn.ModuleList()
        if num_layers>1:
            self.convs.append(
                GCNConv(num_features, hidden_channels))
            for i_layer in range(num_layers-2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
            num_features = hidden_channels
        if not self.lin:
            self.convs.append(
                GCNConv(num_features, num_classes))
        else:
            self.convs.append(
                GCNConv(num_features, hidden_channels))
            
        self.dropout = dropout
        self.norm = norm
        
    
    def forward(self, x, edge_index,**kwargs):
        x_out_list = []
        # print('---start')
        t0 = time.time()
        for i,conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            
            # print('--pure-conv-start-dummy,',1000*(time.time()-t0))
            # print((conv.in_channels,conv.out_channels))
            t0 = time.time()
            x = conv(x, edge_index)
            # print('--pure-conv-end-real,',1000*(time.time()-t0))
            t0 = time.time()
            if i!=len(self.convs)-1 or self.lin:
                x = F.relu(x)
                if self.norm:
#                     x_u = x.mean()
#                     x_sigma = x.std()
#                     x = (x-x_u)/x_sigma
                    rowsum = torch.div(1.0, torch.sum(x, dim=1))
                    rowsum[torch.isinf(rowsum)] = 0.
                    x = torch.mm(torch.diag(rowsum), x)
            x_out_list.append(x)#-2,...
            # print('---conv-end,',1000*(time.time()-t0))
            t0 = time.time()

        if self.lin:
            x = self.lins(x)
        temp = kwargs['temp']
        if temp!=1:
            x=x/temp
        distr = x.log_softmax(dim=-1)
        x_out_list.append(distr)#-1
        # print('--------end,',1000*(time.time()-t0))
        
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
dataset_class = Planetoid(dspath, dataset, split=split_type[i_s], 
                          transform=T.NormalizeFeatures())
data = dataset_class[0]

print(data)
print("train,val,test: ",data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
print("calss_num: ",dataset_class.num_classes)
data = data.cuda()
data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                          sparse_sizes=(data.x.shape[0], data.x.shape[0])).t()
data.adj_t = gcn_norm(data.adj_t)

if args.teacher_name is not None:
    # if args.dataset == 'cora':
    #     dataset_prefix = ''
    # else:
    dataset_prefix = args.dataset+'-'
    with open(f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_distr_run-'+args.teacher_name+".d",'rb') as f:
        distill_label = pickle.load(f).detach()
        if args.temp!=1:
        # import pdb;pdb.set_trace()
            distill_label = (distill_label/args.temp).log_softmax(dim=-1)
    if args.distill_range == 'train':
        distill_mask = data.train_mask
    elif args.distill_range == 'most':
        distill_mask = ~data.test_mask
    elif args.distill_range == 'all':
        distill_mask = (data.train_mask==data.train_mask)

all_train = []
all_val = []
all_test = []
other_all_train = {}
other_all_val = {}
other_all_test = {}
for str_ep in ['200','400','600','800']:
    other_all_train[str_ep] = []
    other_all_val[str_ep] = []
    other_all_test[str_ep] = []
for _ in range(args.runs):

    gcn_model = GCN(hidden_channels=args.hidden_channel, num_layers=args.layer,dropout=args.dropout,
                        num_features=data.x.shape[1],num_classes=dataset_class.num_classes,
                    norm=False,lin=args.lin).cuda()


    gcn_optimizer = torch.optim.Adam(
        gcn_model.parameters(), weight_decay=args.weight_decay, lr=0.01
    )


    total_ep = args.epochs
    log_ep = args.log_ep
    eval_save = args.save_teacher
    best_val = 0
    best_ep = -1
    best_eval = True
    print_train = False

    gcn_train_list = []
    gcn_train_loss_list = []
    gcn_val_list = []
    gcn_test_list = []
    # seed_everything(SEED)

    rebuttal_time = []

    for ep in range(total_ep):
        ep=ep+1
        gcn_model.train()
        gcn_optimizer.zero_grad()
        gcn_distr, gcn_x_out_list= gcn_model(data.x,data.edge_index,temp=args.temp)
        # print(gcn_model)
        # import pdb;pdb.set_trace()
        if args.teacher_name is not None:
            loss = kl(gcn_distr[distill_mask], distill_label[distill_mask])
        else:
            loss = nll(gcn_distr[data.train_mask], data.y[data.train_mask])
        loss.backward()
        gcn_optimizer.step()
        gcn_train_loss_list.append(loss.item())
        with torch.no_grad():
            gcn_model.eval()

            t1 = time.time()
            gcn_distr, gcn_x_out_list= gcn_model(data.x,data.edge_index,temp=1)
            t2 = time.time()
            rebuttal_time.append(t2-t1)
            t += t2-t1
            # tt += 1
            # if tt%100==0:
            #     print(t/tt,'s')
            #     import pdb;pdb.set_trace()

            pred = gcn_distr.max(1)[1]
            right_pred = pred.eq(data.y)
            mask = data.train_mask
            gcn_train_list.append(right_pred[mask].sum().item() / mask.sum().item())
            mask = data.val_mask
            gcn_val_list.append(right_pred[mask].sum().item() / mask.sum().item())
            mask = data.test_mask
            gcn_test_list.append(right_pred[mask].sum().item() / mask.sum().item())
            if str(ep) in ['200','400','600','800']:
                other_all_train[str(ep)].append(gcn_train_list[-1])
                other_all_val[str(ep)].append(gcn_val_list[-1])
                other_all_test[str(ep)].append(gcn_test_list[-1])

        if gcn_val_list[-1]>best_val:
            best_val = gcn_val_list[-1]
            best_ep = ep
            torch.save(gcn_model.state_dict(),args.tmp_model_name)
        if ep%log_ep==0:
            print(f'Epoch: {ep:03d}, Train: {gcn_train_list[-1]:.4f}, \
                Val: {gcn_val_list[-1]:.4f}, Test: {gcn_test_list[-1]:.4f}')

    rebuttal_time_np = np.array(rebuttal_time)
    print("mean:",rebuttal_time_np.mean()*1000)

    if print_train:
        plt.plot(gcn_train_list,c='b',alpha=0.3)
        plt.plot(gcn_val_list,c='r',alpha=0.3)
        plt.plot(gcn_test_list,c='g',alpha=0.3)
    if best_eval:
        gcn_model.load_state_dict(torch.load(args.tmp_model_name))
        gcn_distr, gcn_x_out_list= gcn_model(data.x,data.edge_index,temp=1)

        pred = gcn_distr.max(1)[1]
        right_pred = pred.eq(data.y)
        mask = data.train_mask
        gcn_train_list.append(right_pred[mask].sum().item() / mask.sum().item())
        mask = data.val_mask
        gcn_val_list.append(right_pred[mask].sum().item() / mask.sum().item())
        mask = data.test_mask
        gcn_test_list.append(right_pred[mask].sum().item() / mask.sum().item())
        all_train.append(gcn_train_list[-1])
        all_val.append(gcn_val_list[-1])
        all_test.append(gcn_test_list[-1])
        print(f'Best ep:{best_ep}:{best_val},Train: {gcn_train_list[-1]:.4f}, \
            Val: {gcn_val_list[-1]:.4f}, Test: {gcn_test_list[-1]:.4f}')


print("BESTVAL train:",f"{np.mean(all_train):.4f}, {np.std(all_train):.4f}",
        "; val:",f"{np.mean(all_val):.4f}, {np.std(all_val):.4f}", 
        "; test:",f"{np.mean(all_test):.4f}, {np.std(all_test):.4f}")

print(all_test)

for str_ep in ['200','400','600','800']:
    print(f" EPCH{str_ep} train:",f"{np.mean(other_all_train[str_ep]):.4f}, {np.std(other_all_train[str_ep]):.4f}",
        "; val:",f"{np.mean(other_all_val[str_ep]):.4f}, {np.std(other_all_val[str_ep]):.4f}", 
        "; test:",f"{np.mean(other_all_test[str_ep]):.4f}, {np.std(other_all_test[str_ep]):.4f}")