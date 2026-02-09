import os
import json
import argparse
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_fold_accuracies(data, out_dir):
    vals = data.get('fold_best_vals', [])
    epochs = data.get('fold_best_epochs', [])
    folds = list(range(1, 1 + len(vals)))

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.bar(folds, vals, color='tab:blue')
    plt.ylim(0,1)
    plt.xlabel('Fold')
    plt.ylabel('Best validation accuracy')
    plt.title('Per-fold best validation accuracies')
    plt.grid(axis='y', alpha=0.3)
    out1 = os.path.join(out_dir, 'fold_best_accuracies.png')
    plt.savefig(out1, bbox_inches='tight')
    plt.close()

    if epochs and len(epochs)==len(vals):
        plt.figure(figsize=(6,4))
        plt.scatter(epochs, vals, c='tab:orange')
        for i,(e,v) in enumerate(zip(epochs,vals), start=1):
            plt.annotate(f'F{i}', (e,v))
        plt.xlabel('Epoch of best val')
        plt.ylabel('Best validation accuracy')
        plt.title('Best accuracy vs epoch (per-fold)')
        plt.grid(alpha=0.3)
        out2 = os.path.join(out_dir, 'fold_epoch_vs_acc.png')
        plt.savefig(out2, bbox_inches='tight')
        plt.close()

    return [os.path.join(out_dir, 'fold_best_accuracies.png'), os.path.join(out_dir, 'fold_epoch_vs_acc.png')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='final_cv/results_L2_H128_lr0.001_wd0.0005.json')
    parser.add_argument('--out', type=str, default='figures')
    args = parser.parse_args()

    data = load_json(args.json)
    imgs = plot_fold_accuracies(data, args.out)
    print('Saved images:', imgs)


if __name__ == '__main__':
    main()
