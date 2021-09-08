# Graph Neural Networks with Pytorch
# Target: SIGN: Scalable Inception Graph Neural Networks
# Original Paper: https://arxiv.org/abs/2004.11198
# Original Code: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/sign.py

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import DataLoader

from torch_geometric.datasets import Flickr
import torch_geometric.transforms as T

from torch_sparse import SparseTensor

import pytorch_lightning as pl

sys.path.append('../')
from utils import *
from torch_custom_funcs import *
logger = make_logger(name='sign_logger')


# 1. Preparing Data
# path = os.path.join(os.getcwd(), 'data', 'Flickr')
# transform = T.Compose([T.NormalizeFeatures(), T.SIGN(K)])
# dataset = Flickr(path, transform=T.NormalizeFeatures())
# data = dataset[0]

# Check
# inspector = GraphInspector(data)
# inspector.get_basic_info()
# inspector.inspect('edge_index')
# 'num_nodes': 89250, 'num_edges': 899756, 'num_classes': 7

GET_SIGN_REPORT = False
if GET_SIGN_REPORT:
    row, col = data.edge_index
    adj_t = SparseTensor(
        row=col, col=row, sparse_sizes=(data.num_nodes, data.num_nodes))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    a1 = adj_t
    a2 = a1 @ a1
    a3 = a2 @ a1
    print(a1.density(), a2.density(), a3.density())
    # 0.01% -> 0.97% -> 11.14%

    # Ax
    x1 = adj_t @ data.x
    x2 = adj_t @ x1
    x3 = adj_t @ x2
    x4 = adj_t @ x3

    # Cosine Sim
    x_list = [x1, x2, x3, x4]

    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    N = 4
    output = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            output[i, j] = cos(x_list[i], x_list[j]).detach().mean().item()

    print(pd.DataFrame(output))


# 2. Applying SIGN & Wrapping with DataLoader
# environments
K = 6
BATCH_SIZE = 1024

path = os.path.join(os.getcwd(), 'data', 'Flickr')
transform = T.Compose([T.NormalizeFeatures(), T.SIGN(K)])
dataset = Flickr(path, transform=transform)
data = dataset[0]

device = get_device()

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

train_loader = DataLoader(train_idx, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_idx, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_idx, batch_size=BATCH_SIZE)


# 3. Model & Lightning Module
class GNN(torch.nn.Module):
    def __init__(self, hidden_size, drop_rate):
        super(GNN, self).__init__()

        self.hidden_size = hidden_size
        self.drop_rate = drop_rate

        # MLP for downstream tasks
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for _ in range(K+1):
            self.lins.append(Linear(dataset.num_node_features, hidden_size))
            self.bns.append(nn.BatchNorm1d(num_features=hidden_size))

        self.linear_final = Linear((K+1)*hidden_size, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, xs):
        hs = []
        for i, x in enumerate(xs):
            h = self.lins[i](x)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.drop_rate, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.linear_final(h)
        return h.log_softmax(dim=-1)


# model = GNN(512, 0.5)
# param_report = get_parameter_report(model)
# print(param_report)
# print(get_num_params(model))


class TorchGraph(pl.LightningModule):
    def __init__(self, device):
        super(TorchGraph, self).__init__()
        self.model = GNN(hidden_size=HIDDEN_SIZE, drop_rate=DROP_RATE).to(device)

    def forward(self, idx):
        self.model.eval()

    def training_step(self, batch, batch_idx):
        # batch = idx
        self.model.train()

        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device) for i in range(1, K+1)]
        y_true = data.y[batch].to(device)

        y_pred = self.model(xs)
        loss = F.nll_loss(y_pred, y_true)

        self.log(name="train_loss", value=loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        xs = [data.x[batch].to(device)]
        xs += [data[f'x{i}'][batch].to(device) for i in range(1, K+1)]
        y_true = data.y[batch].to(device)

        y_pred = self.model(xs)
        loss = F.nll_loss(y_pred, y_true)

        self.log(name="val_loss", value=loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return optimizer


# 4. Train
HIDDEN_SIZE = 1024
DROP_RATE = 0.5
LR = 0.01
EPOCHS = 200

sign_graph = TorchGraph(device=device)
logger.info(f"num model parameters: {get_num_params(sign_graph.model)}")
print(sign_graph.model)


trainer = pl.Trainer(
    gpus=1,
    auto_scale_batch_size=None,
    deterministic=True,
    max_epochs=EPOCHS
)

trainer.fit(model=sign_graph, train_dataloader=train_loader, val_dataloaders=val_loader)


# 5. Evaluate
@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = total_examples = 0
    for idx in loader:
        xs = [data.x[idx].to(device)]
        xs += [data[f'x{i}'][idx].to(device) for i in range(1, K + 1)]
        y = data.y[idx].to(device)

        out = model(xs)
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += idx.numel()

    return total_correct / total_examples

train_acc = test(sign_graph.model.to(device), train_loader)
val_acc = test(sign_graph.model.to(device), val_loader)
test_acc = test(sign_graph.model.to(device), test_loader)

print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')




