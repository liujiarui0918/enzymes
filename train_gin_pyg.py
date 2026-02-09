"""
PyG-based GIN training script.
Requires PyTorch Geometric and its dependencies to be installed.

This script provides options to use SAGPool or Set2Set readout.
Usage (example):
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-<your-torch-version>.html
  python train_gin_pyg.py --pool sag --layers 4 --hidden 128 --epochs 200

Note: installation of PyG depends on your CPU/CUDA setup; see https://pytorch-geometric.readthedocs.io/
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GINConv, global_add_pool, SAGPooling, Set2Set
from torch import nn
import networkx as nx
import train_gin as tg


def graphs_to_pyg_dataset(graphs):
    data_list = []
    for G in graphs:
        nodes = list(G.nodes(data=True))
        x = torch.tensor(np.vstack([n[1]['feature'] for n in nodes]), dtype=torch.float)
        # edges
        A = nx.to_numpy_array(G, nodelist=[n[0] for n in nodes]).astype(int)
        src, dst = np.where(A > 0)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        y = torch.tensor([G.graph['label']], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    class ENZDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__('.')
            self.data, self.slices = self.collate(data_list)
    return ENZDataset(data_list)


class PyGGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, pool='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                nn_lin = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            else:
                nn_lin = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn_lin)
            self.convs.append(conv)
            self.batchnorms.append(nn.BatchNorm1d(hidden_dim))
        self.pool = pool
        if pool == 'set2set':
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
            self.lin = nn.Linear(2 * hidden_dim, num_classes)
        else:
            self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.batchnorms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        if self.pool == 'set2set':
            g = self.set2set(x, batch)
        else:
            g = global_add_pool(x, batch)
        return self.lin(g)


def train(model, loader, opt, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, correct / total


def test(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item() * data.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.num_graphs
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', choices=['sum', 'set2set', 'sag'], default='sum')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), 'raw', 'GraphRNN', 'dataset', 'ENZYMES')
    graphs = tg.load_enzymes(data_root)
    dataset = graphs_to_pyg_dataset(graphs)
    # simple split
    idx = np.random.permutation(len(dataset))
    split = int(0.8 * len(dataset))
    train_idx = idx[:split]
    val_idx = idx[split:]
    train_ds = dataset[train_idx.tolist()]
    val_ds = dataset[val_idx.tolist()]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyGGIN(in_dim=dataset[0].x.size(1), hidden_dim=args.hidden, num_layers=args.layers, num_classes=6, pool=args.pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for ep in range(args.epochs):
        tr_loss, tr_acc = train(model, train_loader, opt, device)
        val_loss, val_acc = test(model, val_loader, device)
        if (ep + 1) % 10 == 0:
            print(f'Epoch {ep+1} | tr_acc {tr_acc:.4f} val_acc {val_acc:.4f}')

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f'pyg_model_pool_{args.pool}_L{args.layers}.pth'))


if __name__ == '__main__':
    main()
