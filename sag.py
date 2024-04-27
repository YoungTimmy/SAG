import argparse
import os.path as osp
import os
import sys
import time
import numpy as np
import random
import scipy.sparse as sp
import scipy as sc
from collections import Counter
from sklearn.metrics import f1_score,accuracy_score


import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj, k_hop_subgraph, dense_to_sparse

from model import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--lamb', type=float, default=0.3)
parser.add_argument('--theta', type=float, default=0.05)
parser.add_argument('--e_interval', type=int, default=1)
parser.add_argument('--node_per_class', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
print(args)

if not os.path.exists(f'log/{args.dataset}'):
    os.mkdir(f'log/{args.dataset}')
res_file = open(f'log/{args.dataset}/{args.dataset}_{time.strftime("%Y_%m_%d")}.txt', 'w+')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(root='./data', name='{}'.format(args.dataset),transform=T.NormalizeFeatures())
    data = dataset[0]


NC = dataset.num_classes
nodes_for_train = args.node_per_class*NC
budget = [args.node_per_class]*NC

#remove self loops and make undirected
data.edge_index = to_undirected(data.edge_index)
data.edge_index,_ = remove_self_loops(data.edge_index)


# # Set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# import scipy.sparse as sp
def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

#compute the topology-influence in 2-hop range
edge_index = data.edge_index
adj = to_dense_adj(edge_index).squeeze(0).numpy()
adj = aug_normalized_adjacency(adj)
adj_matrix = torch.FloatTensor(adj.todense())
aadj = torch.mm(adj_matrix,adj_matrix)
inf = aadj

#compute 2-hop edge_index
edge_index2, _ = dense_to_sparse(aadj)


def train(model,optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    results={}
    all_mask = {'train':train_mask,'val':val_mask,'test':test_mask}
    for key in ['train', 'val', 'test']:
        mask = all_mask[key]
        loss = F.cross_entropy(out[mask], data.y[mask])
        acc = accuracy_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy())
        macrof1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro') 
        results['{}_loss'.format(key)] = loss
        results['{}_acc'.format(key)] = acc
        results['{}_macro_f1'.format(key)] = macrof1
    return results


@torch.no_grad()
#calculate the percentage of elements smaller than the k-th element
def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

@torch.no_grad()
def select_nodes(model,ignore_nodes,lamb=0.5):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    out = out-torch.mean(out,dim=0) #shift center
    
    #semantic similarity
    row,col = edge_index2
    # print(f"edge_index2:{edge_index2[:,:100]}")
    similarity = torch.cosine_similarity(out[row], out[col], dim=1)
    sim_matrix = to_dense_adj(edge_index=edge_index2,edge_attr=similarity).squeeze(0)
    
    inf_score = (inf*sim_matrix)

    # compute the increment:
    coverset, e2, mapping, edge_mask = k_hop_subgraph(node_idx=idx_train,num_hops=2,edge_index=data.edge_index)
    # print(coverset)
    inf_score[:,coverset]=-1 #filter the coverset
    increment = torch.count_nonzero(inf_score>args.theta,dim=1).cpu().numpy()

    inf_perc = np.asarray([perc(increment,i) for i in range(len(inf_score))])
    # print(f"inf_perc:{inf_perc}")
    
    #compute the prototype to computer disperity
    xp=[]
    for i in range(dataset.num_classes):
        xp.append(out[idx_train_list[i]].mean(dim=0))
    xpt=torch.stack(xp,dim=0)
    xp_dis = (1-torch.mm(out,xpt.T).max(-1).values).cpu().numpy()
    dis_perc = np.asarray([perc(xp_dis,i) for i in range(len(xp_dis))])
    # print(f"dis_perc:{dis_perc}")
    
    # total score
    finalweight = (1-lamb)*inf_perc + lamb*dis_perc
    finalweight[ignore_nodes]=-1e10
    
    # class-balanced select, select nc nodes in each round
    select = np.argsort(-finalweight)
    has_selected = 0
    for index, idx in enumerate(select):
        predid = pred[idx]
        if budget[predid]>0:
            print(f"select node {idx}")
            idx_train_list[y[idx]].append(idx)
            budget[y[idx]] -= 1 
            has_selected += 1 # keep select nc nodes in each round
            
        if has_selected == NC:
            print("finish selection in this round")
            break
        if sum(budget)==0:
            print("all selection finished")
            break
        if index==data.num_nodes-1:
            print(f"there is no enough nodes for class selection")
    return idx_train_list


random.seed(args.seed)
seeds = random.sample(range(0,10001),10)
print(f"with initial seed:{args.seed}, the sample seeds are {seeds}",file=res_file)

splits_acc = []
splits_macro = []
for seed_id, seed in enumerate(seeds):
    # if seed_id>0: break
    set_random_seed(seed)
    
    data = data.cpu()
    y = data.y
    labels = torch.eye(dataset.num_classes)[y,:]

    idx_test = [i for i, x in enumerate(data.test_mask) if x]
    idx_val = [i for i, x in enumerate(data.val_mask) if x]

    idx_traincand = list(set(range(0, data.num_nodes))-set(idx_val)-set(idx_test))

    traincand_mask = torch.zeros_like(data.y,dtype=bool)
    traincand_mask[idx_traincand]=True

    # sample 4 nodes per class as inital train set
    idx_train_list = []
    for i in range(dataset.num_classes):
        clsi = (data.y==i)&traincand_mask
        nodeidx = torch.where(clsi==True)[0].numpy()
        ridx = random.sample(range(0,nodeidx.shape[0]), 4)       
        idx_train_list.append(list(nodeidx[ridx]))
    # print(idx_train_list) # n_class*4

    budget = [args.node_per_class-4]*dataset.num_classes
    idx_train = []
    for train_nodes in idx_train_list:
        idx_train += train_nodes

    init_idx_train = idx_train
    print(f"idx_train_{len(idx_train)}: {idx_train}",file=res_file)

    # Train mask, val mask and test mask
    train_mask = torch.zeros_like(data.test_mask)
    train_mask[idx_train] = True

    val_mask = data.val_mask
    test_mask = data.test_mask

    model = GCN(dataset.num_features, 128, dataset.num_classes,dropout=args.dropout)
    model, data = model.to(device), data.to(device)
    edge_index2, inf = edge_index2.to(device), inf.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=5e-4)



    best_val_acc = 0
    test_acc = 0
    best_val_macro = 0
    test_macro = 0
    
    best_val_loss = float('inf')
    val_loss_history = []
    for epoch in range(args.epochs):
        loss = train(model,optimizer)
        eval_info = test(model)
        eval_info['epoch']=epoch
        
        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']
            test_macro = eval_info['test_macro_f1']

        val_loss_history.append(eval_info['val_loss'])

        if epoch%args.e_interval==0:
            print(f"Epoch={epoch}, Loss={loss}, train_acc={eval_info['train_acc']}, train_macro_f1={eval_info['train_macro_f1']}, "
            f"val_acc={eval_info['val_acc']}, val_macro_f1={eval_info['val_macro_f1']},"
            f"test_acc={eval_info['test_acc']},test_macro_f1={eval_info['test_macro_f1']}")
            
            if(len(idx_train) < nodes_for_train):
                # print(f"before select:{sum(train_mask)}")
                ignore_nodes = idx_train+idx_val+idx_test
                idx_train_list = select_nodes(model,ignore_nodes=ignore_nodes,lamb=args.lamb)
                # print(idx_train_list)
                idx_train = []
                for train_nodes in idx_train_list:
                    idx_train += train_nodes
                train_mask[idx_train]=True
                print(f"idx_train_{len(idx_train)}: {idx_train}",file=res_file)
                

        if len(idx_train) >= nodes_for_train and epoch>args.epochs//2:
            tmp = torch.tensor([val_loss_history[-(args.early_stopping+1):-1]])
            if eval_info['val_loss'] > tmp.mean().item():
                print("Early stopping...",file=res_file)
                break
    splits_acc.append(test_acc)
    splits_macro.append(test_macro)

    print('---------------------------------------------------------',file=res_file)
    print(f"split {seed_id+1} finished, the test accuracy is {test_acc}, the macro_f1 is {test_macro}",file=res_file)
    print(f"the selected nodes are {idx_train}",file=res_file)
    print('---------------------------------------------------------',file=res_file)


    
accs = torch.tensor(splits_acc)
macros = torch.tensor(splits_macro)
print('---------------------------------------------------------',file=res_file)
print(f"runs finished!",file=res_file)
print(f"mean accuracy: {accs.mean().item():.4f} ± {accs.std().item():.4f}",file=res_file)
print(f"mean macro f1: {macros.mean().item():.4f} ± {macros.std().item():.4f}",file=res_file)
print('---------------------------------------------------------',file=res_file)

print('---------------------------------------------------------')
print(f"runs finished!")
print(f"mean accuracy: {accs.mean().item():.4f} ± {accs.std().item():.4f}")
print(f"mean macro f1: {macros.mean().item():.4f} ± {macros.std().item():.4f}")
print('---------------------------------------------------------')
