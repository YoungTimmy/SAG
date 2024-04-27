import sys
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from model import GCN

# Set random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Settings
parser = argparse.ArgumentParser(description='AGE')
parser.add_argument('--validation_id', default=0, help='The validation set id.')
parser.add_argument('--num_per_class', default=4, type=int, help='Number of the initial labelled nodes per class,')
parser.add_argument('--class_nb', default=7, type=int, help='Number of class for each dataset')
parser.add_argument('--dataset', default='Cora', help='Dataset string.')   # 'Cora', 'Citeseer', 'Pubmed'
parser.add_argument('--model', default='gcn', help='Model string.')  # 'gcn', 'gcn_cheby', 'dense'
parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to train.')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', default=10, type=int, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', default=3, type=int, help='Maximum Chebyshev polynomial degree.')

args = parser.parse_args()

# Dataset
dataset_str = args.dataset
dataset = Planetoid(root='./data', name=dataset_str)
exit()


# Time sensitive weight
basef = 0
if dataset_str == 'Citeseer':
    basef = 0.9
elif dataset_str == 'Cora':
    basef = 0.995
elif dataset_str == 'Pubmed':
    basef = 0.995
if(basef==0):
    print('Error! Have to set basef first!')
    sys.exit()

# Budget, 20 nodes for every class 
NCL = args.class_nb
NL = NCL*20

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(args.epochs):

    # Time sensitive parameters
    gamma = np.random.beta(1, 1.005-basef**epoch)
    alpha = beta = (1-gamma)/2

    # Start
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
