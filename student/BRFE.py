# USAGE EXAPMLES:
# 
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python BRFE.py --dataset cora --layer 2  --nx_num 2 --n_input --n_cvae x_hop1_h[-3] --g_input --near_hop 2 --g_opti -dl 1433 64 64 -hl 64 64 64 --plus gnn-add
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python BRFE.py --dataset citeseer --layer 2  --nx_num 2 --n_input --n_cvae x_hop1_h[-3] --g_input --near_hop 2 --g_opti --g_init -dl 3703 256 -hl 128 128 --plus add-gnn-par --teacher_name None --alpha 0.3
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python BRFE.py --dataset pubmed --layer 2  --nx_num 2 --n_input --n_cvae x_hop1_h[-3] --g_input --near_hop 2 --g_opti --g_init -dl 500 256 -hl 97 97 --plus add-gnn-add --tmp_model_name 0 --alpha 0

######

### import ###
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GCNConv

import argparse
import pickle
import numpy as np
import random
import os
import time
t = 0
tt = 0
######

### parser ###
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help=' ')
parser.add_argument('--h_teacher_name', type=str, default='9', help=' ')
parser.add_argument('--teacher_name', type=str, default='9', help=' ')
parser.add_argument('--distill_range', type=str, default="train", help=' ')

parser.add_argument('--layer', type=int, default=2, help=' ')
parser.add_argument('--mlp_layer', type=int, default=None, help=' ')
parser.add_argument('-dl','--dim_list', nargs='+', help=' ')
parser.add_argument('-hl','--hidden_list', nargs='+', help=' ')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout2', type=float, default=0.9, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help=' ')
parser.add_argument('--alpha', type=float, default=0.5, help=' ')

parser.add_argument('--input_cat_cvae', type=str, default='None1', help='input concate cvae. ')
parser.add_argument('--two_cvae', action='store_true', help='two cvae. ')
parser.add_argument('--cvae_cat', action='store_true', help='concate two cvae. ')
parser.add_argument('--cvae_par', action='store_true', help='parallel two cvae. ')
parser.add_argument('--n_input', action='store_true', help='(work when not) cvae_cat,cvae_par')

parser.add_argument('--g_input', action='store_true', help='add input from graph-level est. (work when not) cvae_cat,cvae_par')
parser.add_argument('--near_hop', type=str, default='None', help=' declare from where to calculate (mean,variance) for (graph-level est.) as the ')
parser.add_argument('--near_far_overlap', type=int, default=0, help=' fix near_hop ')
parser.add_argument('--g_init', action='store_true', help='init mu,sigma as calculated ')
parser.add_argument('--g_opti', action='store_true', help='update the extra input distribution ')

parser.add_argument('--plus', type=str, default='None', help='[gnn-par, gnn-add, idn-add] ')

parser.add_argument('--lam', type=float, default=1., help='The strength of consist loss')
parser.add_argument('--bayes', type=float, default=1., help='The strength of bayes loss')
parser.add_argument('--n_cvae', type=str, default='None', help=' hop1_h[-3],hop2_h[-5],hop(i)_h[(j)]: i means with whom to construct pair for cvae train; j means take what embedding of that who')

parser.add_argument('--seed', type=int, default=1, help=' ')
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lr', type=float, default=0.01, help=' ')
parser.add_argument('--g_lr', type=float, default=0.001, help=' ')
parser.add_argument('--epochs', type=int, default=1000, help=' ')
parser.add_argument('--nx_num', type=int, default=1, help=' ')
parser.add_argument('--sample_num', type=int, default=4, help=' ')
parser.add_argument('--tmp_model_name', type=str, default='tmp', help=' ')
parser.add_argument('--no_200ep_output', action='store_true', help=' ')
parser.add_argument('--log_ep', type=int, default=200, help=' ')

args = parser.parse_args()

dim_list = []
for dn in args.dim_list:
    dim_list.append(eval(dn))
print(f"dim_list {dim_list}")
hidden_list = []
for hn in args.hidden_list:
    hidden_list.append(eval(hn))
print(f"hidden_list {hidden_list}")
# 1.x 2.g 3.n ; use 0 holdplace

######

### random seed ###
SEED = args.seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
seed_everything(SEED)

######

### define class ###

class Gbrfe(nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.mu = nn.Embedding(1,dim)
        self.sigma = nn.Embedding(1,dim)

class BRFEGNN(nn.Module):
    def __init__(self, dim_list, hd_list, nclass, dropout, dropout2, layer_num = 2, plus="None"):
    # def __init__(self, concat, nfeat, nhid, nclass, dropout, dropout2, layer_num = 2, nh_dim=0):
        super().__init__()
        self.plus = plus
        for i in dim_list:
            if i==0:
                raise ValueError
        hd_total = 0
        for i in hd_list:
            hd_total += i
            if i==0:
                raise ValueError
        concat = len(hd_list)
        if plus in ['gnn-add','idn-add','add-gnn-add']:
            hd_total = hd_list[0]

        hd = 4
        self.gcn1_list = nn.ModuleList()
        for i in range(concat):
            if self.plus in ['idn-add'] and i==concat-1:
                self.gcn1_list.append(nn.Identity())
            else:
                self.gcn1_list.append(GCNConv(dim_list[i],hd_list[i]))
                # self.gcn1_list.append(GATConv(dim_list[i],hd_list[i]//hd,hd,dropout=0.))
        if layer_num==1:
            self.lin1 = nn.Linear(hd_total, nclass)
        elif layer_num==2:
            self.gcn2 = GCNConv(hd_total, nclass)
            # self.gcn2 = GATConv(hd_total, nclass,concat=False,dropout=0.)
        elif layer_num == 3:
            self.gcn2 = GCNConv(hd_total, hd_total)
            self.gcn3 = GCNConv(hd_total, nclass)
            # self.gcn2 = GATConv(hd_total, hd_total//hd,hd,dropout=0.)
            # self.gcn3 = GATConv(hd_total, nclass,concat=False,dropout=0.)
        self.dropout = dropout
        self.dropout2 = dropout2
        self.layer_num = layer_num
    
    def forward(self, x_list, edge_idx):
        # print('---start')
        # t0 = time.time()
        hidden_list = []
        # import pdb;pdb.set_trace()
        for k,con in enumerate(self.gcn1_list):
            dropout = self.dropout
            if k!=0:
                dropout = self.dropout2
            x = F.dropout(x_list[k], dropout, training=self.training)

            if args.plus in ['idn-add'] and k==len(self.gcn1_list)-1:
                hidden_list.append(con(x))
            else:
                
                # print('--pure-conv-start-dummy,',1000*(time.time()-t0))
                # print((con.in_channels,con.out_channels))
                # t0 = time.time()
                tmp_conv_h = con(x,edge_idx)
                # print('--pure-conv-end-real,',1000*(time.time()-t0))
                # t0 = time.time()
                hidden_list.append(F.relu(tmp_conv_h))

            # print('---conv1-end,',1000*(time.time()-t0))
            # t0 = time.time()

        if self.plus in ['gnn-add','idn-add','add-gnn-add']:
            x = torch.mean(torch.stack(hidden_list),dim=0)
            # print('---possible-mean-end,',1000*(time.time()-t0))
            # t0 = time.time()
        elif self.plus in ['gnn-par','add-gnn-par']:
            x = torch.cat(hidden_list, dim = -1)
            # print('---possible-cat-end,',1000*(time.time()-t0))
            # t0 = time.time()
        # import pdb;pdb.set_trace()
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer_num==1:
            x = self.lin1(x)
        elif self.layer_num==2:
            x = self.gcn2(x, edge_idx)
        elif self.layer_num==3:
            x = F.relu(x)
            x = self.gcn3(x, edge_idx)
        distr = x.log_softmax(dim=-1)
        # print('--------other-conv-end,',1000*(time.time()-t0))
        # t0 = time.time()
        return distr

class LAMLP(nn.Module):
    def __init__(self, dim_list, hd_list, nclass, dropout, dropout2, layer_num = 2):
    # def __init__(self, concat, nfeat, nhid, nclass, dropout, layer_num = 2, nh_dim=0):
        super().__init__()
        dim_total = 0
        for i in dim_list:
            dim_total += i
            if i==0:
                raise ValueError
        hd_total = 0
        for i in hd_list:
            hd_total += i
            if i==0:
                raise ValueError
        concat = len(hd_list)
        self.lins = nn.ModuleList()

        if layer_num==1:
            self.lins.append(nn.Linear(dim_total,nclass))
        else:
            for i in range(concat):
                self.lins.append(nn.Linear(dim_list[i],hidden_list[i]))

        self.lins2 = nn.ModuleList()
        for i in range(layer_num-1):
            if i==layer_num-2:
                self.lins2.append(nn.Linear(hd_total,nclass))
            else:
                self.lins2.append(nn.Linear(hd_total,hd_total))
        self.dropout = dropout
        self.layer_num = layer_num
    
    def forward(self, x_list, edge_idx):
        hidden_list = []
        # import pdb;pdb.set_trace()
        if self.layer_num==1:
            x_list = [torch.cat(x_list, dim = -1)]
        for k,lin in enumerate(self.lins):
            dropout = self.dropout
            if k!=0:
                dropout = self.dropout2
            x = F.dropout(x_list[k], dropout, training=self.training)
            hidden_list.append(F.relu(lin(x)))
        x = torch.cat(hidden_list, dim = -1)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer_num==1:
            pass
        for i,lin in enumerate(self.lins2):
            if i!=len(self.lins2)-1:
                x = F.relu(x)
            x = lin(x)
        distr = x.log_softmax(dim=-1)
        return distr

######


### define cvae ###

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0):
        super().__init__()
        
        if conditional:
            assert conditional_size > 0
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)
        
        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std
        
    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)
        
        return recon_x
    
class Encoder(nn.Module):
    
    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size,
                out_type = None):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            elif out_type=='sigmoid':
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
            else:
                pass

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

######



### load dataset ###
device = torch.device('cuda:0')
# device = torch.device('cpu')

dataset = args.dataset
dspath = "./data/"+dataset
split_type = ["public","full","geom-gcn","random"][0]
dataset_class = Planetoid(dspath, dataset, split=split_type, 
                          transform=T.NormalizeFeatures())
data = dataset_class[0]
print(data)
print("train,val,test: ",data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
print("calss_num: ",dataset_class.num_classes)
data = data.to(device)
data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                          sparse_sizes=(data.x.shape[0], data.x.shape[0])).t()
data.adj_t = gcn_norm(data.adj_t)

######

### load cvae ###
if args.n_cvae != 'None':
    chp_pth = f'{args.n_cvae}.m'
    print(f">>> load cvae: {chp_pth}")
    chp = torch.load(chp_pth)
    cvae = VAE(**chp[0])
    cvae.load_state_dict(chp[1])
    cvae.to(device)
    cvae.eval()
if args.cvae_cat or args.cvae_par:
    if args.input_cat_cvae != 'None':
        chp_pth = f'{args.input_cat_cvae}.m'
        print(f">>> load cvae: {chp_pth}")
        chp = torch.load(chp_pth)
        cat_cvae = VAE(**chp[0])
        cat_cvae.load_state_dict(chp[1])
        cat_cvae.to(device)
        cat_cvae.eval()
######

### load knowledge ###
if args.h_teacher_name != 'None':
    dataset_prefix = args.dataset+'-'
    gcnii_d_names = [f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_distr_run-{args.h_teacher_name}.d',
                    f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_x_out_list_run-{args.h_teacher_name}.d']

    with open(gcnii_d_names[0],'rb') as f:
        gcnii_distr = pickle.load(f)
    with open(gcnii_d_names[1],'rb') as f:
        gcnii_x_out_list = pickle.load(f)
    if args.near_hop != 'None':
        h_idx = eval(args.near_hop)
        h = gcnii_x_out_list[-h_idx-3+args.near_far_overlap]
        h = h.to(device)
    else:
        print("ATTENTION: args.near_hop == 'None'")
else:
    print("ATTENTION: args.h_teacher_name == 'None'")

######

### load logit ###
if args.teacher_name != 'None':
    # if args.dataset == 'cora':
    #     dataset_prefix = ''
    # else:
    dataset_prefix = args.dataset+'-'
    with open(f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_distr_run-'+args.teacher_name+".d",'rb') as f:
        distill_target = pickle.load(f).to(device)
    if args.distill_range == 'train':
        distill_mask = data.train_mask
    elif args.distill_range == 'most':
        distill_mask = ~data.test_mask
    elif args.distill_range == 'all':
        distill_mask = (data.train_mask==data.train_mask)
    # elif args.distill_range == 'middle1':
    #     distill_mask = ~data.test_mask
    #     distill_mask[1000:] = False

######


### prepare train###
nll = torch.nn.NLLLoss()
kl = nn.KLDivLoss(reduction="sum",log_target=True)

######

### train ###

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def accuracy(output, y):
    pred = output.max(1)[1]
    return pred.eq(y).sum()/output.shape[0]

######



### train ###
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

nx_list = []
log_ep = args.log_ep

nx_num = args.nx_num
nx_sample = args.sample_num
epochs = args.epochs

for _ in range(args.runs):

    ### time ###
    time_n_input_rand = []
    time_n_input_cal = []
    time_g_input_rand = []
    time_g_input_cal = []
    time_gnn = []

    
    if args.g_input:
        G_BRFE = Gbrfe(h.shape[1]).to(device)
        mu = torch.mean(h[data.train_mask],dim=0)
        sigma = torch.var(h[data.train_mask],dim=0)
        if args.g_init:
            G_BRFE.mu.weight.data = mu.clone()
            G_BRFE.sigma.weight.data = sigma.clone()
        else:
            G_BRFE.mu.weight.data = torch.zeros(h.shape[1]).to(device)
            G_BRFE.sigma.weight.data = torch.ones(h.shape[1]).to(device)
    if args.layer == 0:
        brfegnn = LAMLP(dim_list=dim_list,
                    hd_list=hidden_list,
                    nclass=dataset_class.num_classes,
                    dropout=args.dropout,
                    dropout2=args.dropout2,
                    layer_num=args.mlp_layer
                    )
    else:
        brfegnn = BRFEGNN(dim_list=dim_list,
                    hd_list=hidden_list,
                    nclass=dataset_class.num_classes,
                    dropout=args.dropout,
                    dropout2=args.dropout2,
                    layer_num=args.layer,
                    plus=args.plus
                    )
    brfegnn.to(device)
    # print(brfegnn)


    brfegnn_optim = torch.optim.Adam(brfegnn.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    if args.g_input:
        gbrfe_optim = torch.optim.Adam(G_BRFE.parameters(),lr=args.g_lr)

    train_list = []
    val_list = []
    test_list = []
    best_val = 0

    for ep in range(epochs):
        brfegnn.train()
        brfegnn_optim.zero_grad()
        if args.g_input:
            gbrfe_optim.zero_grad()
        output_list = []
        for k in range(nx_sample):
            nx_list = []
            #
            if nx_num==1:
                assert args.n_input+args.g_input==1, "nx_num==1,but args.n_input+args.g_input!=1"
                if args.n_input:
                    nx = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                elif args.g_input:
                    nx = torch.randn(data.x.shape[0],h.shape[1]).to(device)
                    nx = G_BRFE.mu.weight+nx*G_BRFE.sigma.weight
                nx_list.append(nx) 
            #
            elif nx_num==2:
                pass
                if args.two_cvae:
                    if args.cvae_cat:
                        nx1 = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx2 = cat_cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx = torch.cat((nx1,nx2),dim=1) # two extra input from cvae, and cat them as one
                        nx_list.append(nx)
                    elif args.cvae_par:
                        nx1 = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx2 = cat_cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx_list.append(nx1)
                        nx_list.append(nx2) # two extra input from cvae, and par them
                else:
                    assert args.n_input+args.g_input==2, "nx_num==2,but args.n_input+args.g_input!=2"
                    nx_n = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                    nx_g = torch.randn(data.x.shape[0],h.shape[1]).to(device)
                    nx_g = G_BRFE.mu.weight+nx_g*G_BRFE.sigma.weight
                    if args.plus in ['gnn-add','gnn-par']:
                        nx_list.append(nx_n)
                        nx_list.append(nx_g)
                    elif args.plus in ['idn-add','add-gnn-add','add-gnn-par']:
                        nx_list.append(args.alpha*nx_n+(1-args.alpha)*nx_g)
            #
            output = brfegnn([data.x]+nx_list,data.edge_index)
            output_list.append(output)
        loss_train = 0.
        
        for k in range(len(output_list)):
            if args.teacher_name == 'None':
                loss_train += nll(output_list[k][data.train_mask],data.y[data.train_mask])
            else:
                loss_train += kl(output_list[k][distill_mask],distill_target[distill_mask])
        loss_train = loss_train/len(output_list)
        loss_bayesian = torch.tensor(0.).to(device)
        if args.g_input:
            if args.bayes>0:
                loss_bayesian += torch.sum(torch.log((sigma+1e-10)/(G_BRFE.sigma.weight+1e-10)+1e-10))-h.shape[1]
                # print(loss_bayesian.item())
                loss_bayesian += torch.sum((G_BRFE.mu.weight-mu)**2/(sigma+1e-10))
                # print((torch.sum((G_BRFE.mu.weight-mu)**2/(sigma+1e-10))).item())
                loss_bayesian += torch.sum((G_BRFE.sigma.weight)/(sigma+1e-10))
                # print((torch.sum((G_BRFE.sigma.weight)/(sigma+1e-10))).item())
                loss_bayesian = loss_bayesian/2
                # print('---')
        if args.lam<=0:
            loss_consis = 0
        else:
            loss_consis =  consis_loss(output_list)
        if ep%log_ep==0:
            # print(f"Ep{ep},loss_train(sample mean):{loss_train.item()},loss_consis:{loss_consis.item()}")
            pass
        # print(loss_train.item())
        # print(loss_consis.item())
        # print(loss_bayesian.item())
        # import pdb;pdb.set_trace()
        loss_train = loss_train + loss_consis + args.bayes * loss_bayesian
        loss_train.backward()
        brfegnn_optim.step()
        if args.g_input:
            if args.g_opti:
                gbrfe_optim.step()
            G_BRFE.sigma.weight.data = F.relu(G_BRFE.sigma.weight.data)
        with torch.no_grad():
            brfegnn.eval()
            nx_list = []
            #
            if nx_num==1:
                assert args.n_input+args.g_input==1, "nx_num==1,but args.n_input+args.g_input!=1"
                if args.n_input:
                    t_st = time.time()
                    tmp_rand = torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device)
                    time_n_input_rand.append(time.time()-t_st)
                    t_st = time.time()
                    nx = cvae.inference(tmp_rand,data.x)
                    time_n_input_cal.append(time.time()-t_st)
                elif args.g_input:
                    t_st = time.time()
                    nx = torch.randn(data.x.shape[0],h.shape[1]).to(device)
                    time_g_input_rand.append(time.time()-t_st)
                    t_st = time.time()
                    nx = G_BRFE.mu.weight+nx*G_BRFE.sigma.weight
                    time_g_input_cal.append(time.time()-t_st)
                nx_list.append(nx) 
            #
            elif nx_num==2:
                pass
                if args.two_cvae:
                    if args.cvae_cat:
                        nx1 = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx2 = cat_cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx = torch.cat((nx1,nx2),dim=1) # two extra input from cvae, and cat them as one
                        nx_list.append(nx)
                    elif args.cvae_par:
                        nx1 = cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx2 = cat_cvae.inference(torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device),data.x)
                        nx_list.append(nx1)
                        nx_list.append(nx2) # two extra input from cvae, and par them
                else:
                    assert args.n_input+args.g_input==2, "nx_num==2,but args.n_input+args.g_input!=2"
                    t_st = time.time()
                    tmp_rand = torch.randn(data.x.shape[0],chp[0]['latent_size']).to(device)
                    time_n_input_rand.append(time.time()-t_st)
                    t_st = time.time()
                    nx_n = cvae.inference(tmp_rand,data.x)
                    time_n_input_cal.append(time.time()-t_st)

                    t_st = time.time()
                    nx_g = torch.randn(data.x.shape[0],h.shape[1]).to(device)
                    time_g_input_rand.append(time.time()-t_st)
                    t_st = time.time()
                    nx_g = G_BRFE.mu.weight+nx_g*G_BRFE.sigma.weight
                    time_g_input_cal.append(time.time()-t_st)

                    if args.plus in ['gnn-add','gnn-par']:
                        nx_list.append(nx_n)
                        nx_list.append(nx_g)
                    elif args.plus in ['idn-add','add-gnn-add','add-gnn-par']:
                        t_st = time.time()
                        nx_list.append(args.alpha*nx_n+(1-args.alpha)*nx_g)
                        t_tmp = time.time()-t_st
                        time_g_input_cal[-1]=time_g_input_cal[-1]+t_tmp/2
                        time_n_input_cal[-1]=time_n_input_cal[-1]+t_tmp/2

            #
            t1 = time.time()
        
            t_st = time.time()
            output = brfegnn([data.x]+nx_list, data.edge_index)
            time_gnn.append(time.time()-t_st)


            t2 = time.time()
            # t += t2-t1
            # tt += 1
            # if tt%100==0:
            #     print(t/tt,'s')
            #     import pdb;pdb.set_trace()

            acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
            acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
            acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
            train_list.append(acc_train)
            val_list.append(acc_val)
            test_list.append(acc_test)
            if str(ep) in ['200','400','600','800']:
                other_all_train[str(ep)].append(train_list[-1].item())
                other_all_val[str(ep)].append(val_list[-1].item())
                other_all_test[str(ep)].append(test_list[-1].item())

        if val_list[-1]>best_val:
            best_val = val_list[-1]
            best_ep_val = best_val.item()
            best_ep_test = test_list[-1].item()
            best_ep_train = train_list[-1].item()
            best_ep = ep
            torch.save(brfegnn.state_dict(),args.tmp_model_name)
            if args.g_input:
                torch.save(G_BRFE.state_dict(),args.tmp_model_name+'gbrfe')

        # if ep%log_ep==0 and ep!=0:
        #     print(f'Epoch: {ep:03d}, Train: {train_list[-1]:.4f}, \
        #         Val: {val_list[-1]:.4f}, Test: {test_list[-1]:.4f}')


    all_train.append(best_ep_train)
    all_val.append(best_ep_val)
    all_test.append(best_ep_test)
    print(f'Best ep:{best_ep}:{best_val:4f},Train: {best_ep_train:.4f}, \
        Val: {best_ep_val:.4f}, Test: {best_ep_test:.4f}')


print("BESTVAL train:",f"{np.mean(all_train):.4f}, {np.std(all_train):.4f}",
        "; val:",f"{np.mean(all_val):.4f}, {np.std(all_val):.4f}", 
        "; test:",f"{np.mean(all_test):.4f}, {np.std(all_test):.4f}")

print(all_test)


if not args.no_200ep_output:
    for str_ep in ['200','400','600','800']:
        print(f" EPCH{str_ep} train:",f"{np.mean(other_all_train[str_ep]):.4f}, {np.std(other_all_train[str_ep]):.4f}",
            "; val:",f"{np.mean(other_all_val[str_ep]):.4f}, {np.std(other_all_val[str_ep]):.4f}", 
            "; test:",f"{np.mean(other_all_test[str_ep]):.4f}, {np.std(other_all_test[str_ep]):.4f}")


######

