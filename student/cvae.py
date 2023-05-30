# USAGE EXAPMLES:
# 
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3 --dataset citeseer
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cvae.py --distill_range train --cvae_type nh-3 --dataset pubmed
# 
# always change the .m output file name to x_hop1_h[-3].m for latter use
# 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt

import pickle
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import os
import gc
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument('--teacher_name', type=str, default = '9', help=' ')
parser.add_argument('--distill_range', type=str, required=True, help=' ')
parser.add_argument('--epochs', type=int, default=1000, help=' ')
parser.add_argument('--cvae_type', type=str, required=True, help=' ')
parser.add_argument('--dataset', type=str, default='cora', help=' ')
parser.add_argument('--seed', type=int, default=2022, help=' ')
args = parser.parse_args()

SEED = args.seed
def seed_everything(seed):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)
torch.use_deterministic_algorithms(True)

#################

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
    
def loss_fn(recon_x, x, mean, log_var):
#     BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     BCE = F.l1_loss(recon_x,x,reduction='sum')
    BCE = F.mse_loss(recon_x,x,reduction='sum')
#     BCE = 0
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     KLD = 0
    return (BCE + KLD) #/ x.size(0)

################



# load cora
dataset = args.dataset
dspath = "./data/"+dataset
split_type = ["public","full","geom-gcn","random"]
i_s = 0
dataset_class = Planetoid(dspath, dataset, split=split_type[i_s], 
                          transform=T.NormalizeFeatures())
data = dataset_class[0]
print(data)
print("train,val,test: ",data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
print("calss_num: ",dataset_class.num_classes)
# device = torch.device('cuda:0')
# device = torch.device('cpu')
data = data.cuda()
data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                          sparse_sizes=(data.x.shape[0], data.x.shape[0])).t()
data.adj_t = gcn_norm(data.adj_t)

# load teacher distr
nll = torch.nn.NLLLoss()
kl = nn.KLDivLoss(reduction="sum",log_target=True)

# load teacher features
# if args.dataset == 'cora':
#     dataset_prefix = ''
# else:
dataset_prefix = args.dataset+'-'
gcnii_d_names = [f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_distr_run-{args.teacher_name}.d',
                 f'../teacher/saved_teacher/{args.dataset}/{dataset_prefix}gcnii_x_out_list_run-{args.teacher_name}.d']

with open(gcnii_d_names[0],'rb') as f:
    gcnii_distr = pickle.load(f)
with open(gcnii_d_names[1],'rb') as f: 
    gcnii_x_out_list = pickle.load(f)



#################
if args.cvae_type == 'nx':
    h = data.x.cpu()
elif args.cvae_type[:2] == 'nh':
    h_idx = eval(args.cvae_type[2:])
    h = gcnii_x_out_list[h_idx].cpu()
if args.distill_range=='train':
    nb_gen_train_id = data.train_mask.nonzero().squeeze().tolist()
elif args.distill_range=='all':
    nb_gen_train_id = range(data.x.shape[0])
elif args.distill_range=='most':
    nb_gen_train_id = (~data.test_mask).nonzero().squeeze().tolist()
elif args.distill_range == 'middle1':
    nb_gen_train_id = ~data.test_mask
    nb_gen_train_id[1000:] = False
    nb_gen_train_id = nb_gen_train_id.nonzero().squeeze().tolist()
    

center_h = False
x_list = []
c_list = []
for i in nb_gen_train_id:
    nb_id = [nid.item() for nid in data.adj_t[i].coo()[1]]
    if not center_h:
        nb_id.remove(i)
    x = h[nb_id]
    c = np.tile(data.x[i].cpu(), (x.shape[0], 1))
#     x = x-c
    x_list.append(x)
    c_list.append(c)
features_x = np.vstack(x_list)
features_c = np.vstack(c_list)
del x_list
del c_list
gc.collect()

features_x = torch.tensor(features_x, dtype=torch.float32)
features_c = torch.tensor(features_c, dtype=torch.float32)


cvae_dataset = TensorDataset(features_x, features_c)
cvae_dataset_sampler = RandomSampler(cvae_dataset)
cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=512)


pretrain_epochs=args.epochs
conditional=True

cvae = VAE(encoder_layer_sizes = [h.shape[1],256],
          latent_size = 100,
        #   decoder_layer_sizes = [256, h.shape[1]],
          decoder_layer_sizes = [64,h.shape[1]],
          conditional = conditional,
          conditional_size = data.x.shape[1])
cvae_optimizer = torch.optim.Adam(cvae.parameters(),lr=0.001)

cvae.cuda()

cvae_loss_list = []
for ep in range(pretrain_epochs):
    # For CVAE, x is input, c is condition
    
    for _,(x,c) in enumerate(cvae_dataset_dataloader):
        cvae_epoch_loss = []
        cvae.train()
        x,c = x.cuda(), c.cuda()
        if conditional:
            recon_x, mean, log_var, _ = cvae(x,c)
        else:
            recon_x, mean, log_var, _ = cvae(x)
        cvae_loss = loss_fn(recon_x, x, mean, log_var)
        
        cvae_optimizer.zero_grad()
        cvae_loss.backward()
        cvae_optimizer.step()
        cvae_epoch_loss.append(cvae_loss.item())
    cvae_loss_list.append(np.mean(cvae_epoch_loss))
#     print(cvae_loss_list[-1])

vae_config = {'encoder_layer_sizes': [h.shape[1],256],
          'latent_size' : 100,
        #   'decoder_layer_sizes' : [256, h.shape[1]],
          'decoder_layer_sizes' : [64,h.shape[1]],
          'conditional' : conditional,
          'conditional_size' : data.x.shape[1]}
if args.distill_range=='all':
    postfix = ''
elif args.distill_range=='train':
    postfix = '-train'
elif args.distill_range=='most':
    postfix = '-most'
elif args.distill_range=='middle1':
    postfix = '-middle1'

cvae_name = f'{args.dataset}_{args.cvae_type}{postfix}_norm_cvae'

if not os.path.exists("./"+cvae_name+"_loss_list"):
    torch.save(cvae_loss_list,"../"+cvae_name+"_loss_list")
    plt.plot(cvae_loss_list)
    print(cvae_loss_list[-10:])
    plt.savefig("./"+cvae_name+"_loss_fig.jpg")

fname = f"./{cvae_name+'.m'}"
if os.path.exists(fname):
    print(f">>>> EXIST: {fname}")
else:
    torch.save([vae_config,cvae.state_dict()],fname)

