import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def load_enzymes(root):
    # expects files in root: ENZYMES_A.txt, ENZYMES_node_attributes.txt,
    # ENZYMES_node_labels.txt, ENZYMES_graph_indicator.txt, ENZYMES_graph_labels.txt
    A = np.loadtxt(os.path.join(root, 'ENZYMES_A.txt'), delimiter=',').astype(int)
    node_att = np.loadtxt(os.path.join(root, 'ENZYMES_node_attributes.txt'), delimiter=',')
    node_label = np.loadtxt(os.path.join(root, 'ENZYMES_node_labels.txt'), delimiter=',').astype(int)
    graph_indicator = np.loadtxt(os.path.join(root, 'ENZYMES_graph_indicator.txt'), delimiter=',').astype(int)
    graph_labels = np.loadtxt(os.path.join(root, 'ENZYMES_graph_labels.txt'), delimiter=',').astype(int)

    # build global graph
    edges = list(map(tuple, A))
    G = nx.Graph()
    G.add_edges_from(edges)
    # add node attributes (nodes are 1-indexed in files)
    for i in range(node_att.shape[0]):
        G.add_node(i + 1, feature=node_att[i], label=int(node_label[i]))
    # remove isolates
    G.remove_nodes_from(list(nx.isolates(G)))

    graphs = []
    num_graphs = int(graph_labels.shape[0])
    node_list = np.arange(graph_indicator.shape[0]) + 1
    for i in range(num_graphs):
        nodes = node_list[graph_indicator == (i + 1)]
        G_sub = G.subgraph(nodes).copy()
        G_sub.graph['label'] = int(graph_labels[i]) - 1  # make 0-based
        graphs.append(G_sub)
    return graphs


class EnzymesDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        G = self.graphs[idx]
        nodes = list(G.nodes(data=True))
        features = np.vstack([n[1]['feature'] for n in nodes]).astype(np.float32)
        # per-graph standardization
        mu = features.mean(axis=0, keepdims=True)
        sigma = features.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        features = (features - mu) / (sigma + 1e-8)
        adj = nx.to_numpy_array(G, nodelist=[n[0] for n in nodes]).astype(np.float32)
        label = G.graph.get('label', 0)
        return features, adj, int(label)


def collate_fn(batch):
    # pad to max nodes in batch
    feats = [torch.from_numpy(b[0]) for b in batch]
    adjs = [torch.from_numpy(b[1]) for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    max_n = max([f.size(0) for f in feats])
    feat_dim = feats[0].size(1)
    batch_feats = torch.zeros(len(batch), max_n, feat_dim)
    batch_adjs = torch.zeros(len(batch), max_n, max_n)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for i, (f, a) in enumerate(zip(feats, adjs)):
        n = f.size(0)
        batch_feats[i, :n, :] = f
        batch_adjs[i, :n, :n] = a
        mask[i, :n] = 1
    return batch_feats, batch_adjs, labels, mask


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x: (B, N, D) or (M, D)
        orig_shape = None
        if x.dim() == 3:
            B, N, D = x.size()
            x = x.view(B * N, D)
            orig_shape = (B, N)
        out = self.net(x)
        if orig_shape is not None:
            B, N = orig_shape
            out = out.view(B, N, -1)
        return out


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0, dropout=0.0):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, dropout=dropout)
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, x, adj):
        # x: (B, N, D), adj: (B, N, N)
        # add self loops
        I = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        adj2 = adj + I
        neigh = torch.matmul(adj2, x)
        out = self.mlp((1 + self.eps) * x + neigh)
        return out


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(GINLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(hidden_dim, hidden_dim, dropout=dropout))
        # classifier will take concatenated pooled outputs from all layers (JK style)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.num_layers = num_layers

    def forward(self, x, adj, mask):
        # x: (B, N, D), adj: (B, N, N), mask: (B, N)
        h = x
        pooled = []
        for layer in self.layers:
            h = layer(h, adj)
            maskf = mask.unsqueeze(-1).float()
            h_masked = h * maskf
            g = h_masked.sum(dim=1)  # (B, D)
            pooled.append(g)
        # concat pooled representations from all layers
        g_all = torch.cat(pooled, dim=1)
        out = self.classifier(g_all)
        return out


def train_one_epoch(model, opt, loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for feats, adjs, labels, mask in loader:
        feats = feats.to(device)
        adjs = adjs.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        opt.zero_grad()
        logits = model(feats, adjs, mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, adjs, labels, mask in loader:
            feats = feats.to(device)
            adjs = adjs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(feats, adjs, mask)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * feats.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += feats.size(0)
    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training & Validation Loss')

    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training & Validation Accuracy')

    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved plot to', out_path)


def plot_compare(all_histories, layer_list, out_path):
    # all_histories: list of dicts with 'train_accs','val_accs','train_losses','val_losses'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist, l in zip(all_histories, layer_list):
        axes[0].plot(hist['val_accs'], label=f'val acc L={l}')
        axes[0].plot(hist['train_accs'], linestyle='--', alpha=0.6, label=f'train acc L={l}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right', fontsize='small')
    axes[0].set_title('Accuracy Comparison')

    for hist, l in zip(all_histories, layer_list):
        axes[1].plot(hist['val_losses'], label=f'val loss L={l}')
        axes[1].plot(hist['train_losses'], linestyle='--', alpha=0.6, label=f'train loss L={l}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].set_title('Loss Comparison')

    plt.tight_layout()
    plt.savefig(out_path)
    print('Saved comparison plot to', out_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=4, help='number of GIN layers')
    parser.add_argument('--hidden-dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--out', type=str, default='training_curves_improved.png', help='output image path')
    parser.add_argument('--compare-layers', type=str, default='',
                        help='comma-separated list of layer counts to compare, e.g. "2,4,6"')
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), 'raw', 'GraphRNN', 'dataset', 'ENZYMES')
    graphs = load_enzymes(data_root)
    # stratified split preserving label distribution
    labels = np.array([g.graph['label'] for g in graphs])
    train_graphs = []
    val_graphs = []
    rng = np.random.RandomState(42)
    for lbl in np.unique(labels):
        inds = np.where(labels == lbl)[0]
        rng.shuffle(inds)
        split = int(0.8 * len(inds))
        train_graphs += [graphs[i] for i in inds[:split]]
        val_graphs += [graphs[i] for i in inds[split:]]

    train_ds = EnzymesDataset(train_graphs)
    val_ds = EnzymesDataset(val_graphs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # safe access to first node's feature vector
    first_node = list(train_graphs[0].nodes(data=True))[0]
    in_dim = first_node[1]['feature'].shape[0]
    # helper to run single experiment for given layer count
    def run_single(layers):
        model = GIN(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=layers, num_classes=6, dropout=args.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val = 0.0
        best_epoch = 0
        for ep in range(args.epochs):
            tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_accs.append(tr_acc)
            val_accs.append(val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = ep + 1
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f'best_model_L{layers}.pth'))
            print(f'Epoch {ep+1:3d} | L={layers} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        history = {'train_losses': train_losses, 'val_losses': val_losses,
                   'train_accs': train_accs, 'val_accs': val_accs,
                   'best_val': best_val, 'best_epoch': best_epoch}
        return history

    # compare mode: run multiple layer configs sequentially and save combined plot
    if args.compare_layers:
        layer_list = [int(x) for x in args.compare_layers.split(',') if x.strip()]
        all_hist = []
        best_overall = {'layers': None, 'val': 0.0, 'epoch': None}
        for l in layer_list:
            print('\nRunning experiment for layers =', l)
            hist = run_single(l)
            all_hist.append(hist)
            if hist['best_val'] > best_overall['val']:
                best_overall = {'layers': l, 'val': hist['best_val'], 'epoch': hist['best_epoch']}
        cmp_out = os.path.join(os.path.dirname(__file__), args.out.replace('.png', '_compare.png'))
        plot_compare(all_hist, layer_list, cmp_out)
        print('Best config:', best_overall)
        return

    # otherwise run single experiment with provided --layers
    model = GIN(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.layers, num_classes=6, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    epochs = args.epochs
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        print(f'Epoch {ep+1:3d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    out = os.path.join(os.path.dirname(__file__), args.out)
    plot_curves(train_losses, val_losses, train_accs, val_accs, out)


if __name__ == '__main__':
    main()
