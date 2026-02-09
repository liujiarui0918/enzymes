import os
import csv
import time
import json
import numpy as np
from itertools import product
 

import torch
from torch.utils.data import DataLoader

import train_gin as tg


def parse_list(s, cast=int):
    return [cast(x) for x in s.split(',') if x.strip()]


def run_grid(data_root, layers_list, hidden_list, lr_list, weight_list, kfold=3, epochs=300, batch_size=16, dropout=0.5, device=None, out_dir=None):
    graphs = tg.load_enzymes(data_root)
    labels = np.array([g.graph['label'] for g in graphs])
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    if out_dir is None:
        out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    combos = list(product(layers_list, hidden_list, lr_list, weight_list))
    def stratified_kfold_indices(labels, n_splits=3, random_state=42):
        rng = np.random.RandomState(random_state)
        labels = np.array(labels)
        folds = [[] for _ in range(n_splits)]
        for cls in np.unique(labels):
            inds = np.where(labels == cls)[0].tolist()
            rng.shuffle(inds)
            for i, idx in enumerate(inds):
                folds[i % n_splits].append(idx)
        # produce (train_idx, val_idx) pairs
        for i in range(n_splits):
            val = np.array(folds[i])
            train = np.array([idx for j, f in enumerate(folds) if j != i for idx in f])
            yield train, val

    for layers, hidden, lr, wd in combos:
        combo_name = f'L{layers}_H{hidden}_lr{lr}_wd{wd}'
        print('\n=== Running combo:', combo_name, '===')
        fold_best_vals = []
        fold_best_epochs = []
        fold_models = []
        fold_idx = 0
        for train_idx, val_idx in stratified_kfold_indices(labels, n_splits=kfold, random_state=42):
            fold_idx += 1
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_ds = tg.EnzymesDataset(train_graphs)
            val_ds = tg.EnzymesDataset(val_graphs)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=tg.collate_fn)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=tg.collate_fn)

            model = tg.GIN(in_dim=list(train_graphs[0].nodes(data=True))[0][1]['feature'].shape[0],
                           hidden_dim=hidden, num_layers=layers, num_classes=6, dropout=dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

            best_val = 0.0
            best_epoch = 0
            best_state = None
            for ep in range(epochs):
                tr_loss, tr_acc = tg.train_one_epoch(model, opt, train_loader, device)
                val_loss, val_acc = tg.evaluate(model, val_loader, device)
                scheduler.step(val_loss)
                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = ep + 1
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if (ep + 1) % 50 == 0:
                    print(f'Combo {combo_name} Fold {fold_idx} Ep {ep+1} | tr_acc {tr_acc:.4f} val_acc {val_acc:.4f}')

            fold_best_vals.append(best_val)
            fold_best_epochs.append(best_epoch)
            if best_state is not None:
                model_path = os.path.join(out_dir, f'best_{combo_name}_fold{fold_idx}.pth')
                torch.save(best_state, model_path)
                fold_models.append(model_path)
            else:
                fold_models.append(None)

        mean_val = float(np.mean(fold_best_vals))
        std_val = float(np.std(fold_best_vals))
        results.append({'combo': combo_name, 'layers': layers, 'hidden': hidden, 'lr': lr, 'weight_decay': wd,
                        'mean_val': mean_val, 'std_val': std_val, 'fold_best_vals': fold_best_vals,
                        'fold_best_epochs': fold_best_epochs, 'fold_models': fold_models})

        # save intermediate results
        with open(os.path.join(out_dir, f'results_{combo_name}.json'), 'w') as f:
            json.dump(results[-1], f, indent=2)

    # save summary CSV
    csv_path = os.path.join(out_dir, 'grid_search_summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['combo', 'layers', 'hidden', 'lr', 'weight_decay', 'mean_val', 'std_val', 'fold_best_vals'])
        writer.writeheader()
        for r in results:
            writer.writerow({'combo': r['combo'], 'layers': r['layers'], 'hidden': r['hidden'], 'lr': r['lr'], 'weight_decay': r['weight_decay'], 'mean_val': r['mean_val'], 'std_val': r['std_val'], 'fold_best_vals': r['fold_best_vals']})

    # choose best overall
    best = max(results, key=lambda x: x['mean_val'])
    print('\n=== Grid search complete. Best combo:', best['combo'], 'mean_val=', best['mean_val'])
    # copy best fold model as best_overall
    best_model_src = best['fold_models'][np.argmax(best['fold_best_vals'])]
    if best_model_src:
        best_model_dst = os.path.join(out_dir, 'best_overall.pth')
        torch.save(torch.load(best_model_src), best_model_dst)
        print('Saved best overall model to', best_model_dst)

    return results, best


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='2,4,6')
    parser.add_argument('--hidden', type=str, default='128')
    parser.add_argument('--lrs', type=str, default='0.001,0.0005')
    parser.add_argument('--wds', type=str, default='0.0005')
    parser.add_argument('--kfold', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--out', type=str, default='grid_results')
    args = parser.parse_args()

    layers_list = parse_list(args.layers, int)
    hidden_list = parse_list(args.hidden, int)
    lr_list = parse_list(args.lrs, float)
    wd_list = parse_list(args.wds, float)

    data_root = os.path.join(os.path.dirname(__file__), 'raw', 'GraphRNN', 'dataset', 'ENZYMES')
    out_dir = os.path.join(os.path.dirname(__file__), args.out)
    start = time.time()
    results, best = run_grid(data_root, layers_list, hidden_list, lr_list, wd_list, kfold=args.kfold, epochs=args.epochs, batch_size=args.batch_size, dropout=args.dropout, out_dir=out_dir)
    print('Total time (s):', time.time() - start)
