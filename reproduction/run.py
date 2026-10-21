"""Usage: python -m reproduction.run --experiment all"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results' / '.mpl-cache'))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from .experiments import himmelblau, small_comparison, mnist


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summary_rows(result):
    return [dict(Optimizer=name,
                 **{'Best Train Loss': min(values['train_loss']),
                    'Best Train Acc (%)': max(values['train_accuracy']),
                    'Best Test Loss': min(values['test_loss']),
                    'Best Test Acc (%)': max(values['test_accuracy'])})
            for name, values in result.items()]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(folder, kind, result):
    save_json(folder / 'metrics.json', result)
    if kind == 'mnist':
        write_csv(folder / 'summary.csv', summary_rows(result))
        rows = [dict(optimizer=name, epoch=i+1,
                     **{key: values[key][i] for key in ['train_loss', 'train_accuracy',
                                                        'test_loss', 'test_accuracy']})
                for name, values in result.items() for i in range(len(values['train_loss']))]
        write_csv(folder / 'epochs.csv', rows)
        keys = ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'batch_loss']
    else:
        keys = ['loss', 'gradient_norm'] if kind == 'himmelblau' else ['loss']
        rows = [dict(optimizer=name, step=i, **{key: values[key][i] for key in keys})
                for name, values in result.items() for i in range(len(values['loss']))]
        write_csv(folder / 'history.csv', rows)
    for key in keys:
        fig, axis = plt.subplots(figsize=(9, 5))
        for name, values in result.items():
            axis.plot(values[key], label=name)
        axis.set(xlabel='Batch' if key == 'batch_loss' else 'Iteration',
                 ylabel=key.replace('_', ' ').capitalize(), title=kind)
        if kind == 'himmelblau':
            axis.set_yscale('log')
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(folder / f'{key}.png', dpi=150)
        plt.close(fig)
    if kind == 'himmelblau':
        fig, axis = plt.subplots(figsize=(7, 6))
        x, y = np.meshgrid(np.linspace(-6, 6, 150), np.linspace(-6, 6, 150))
        axis.contour(x, y, (x*x+y-11)**2+(x+y*y-7)**2, levels=40)
        for name, values in result.items():
            points = np.array(values['positions'] + [values['final_position']])
            axis.plot(points[:, 0], points[:, 1], label=name)
        axis.set(xlabel='x', ylabel='y', title='Himmelblau: actual position snapshots')
        axis.legend()
        fig.tight_layout()
        fig.savefig(folder / 'trajectory.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all', 'himmelblau', 'toy', 'tiny-mnist', 'mnist'], default='all')
    parser.add_argument('--epochs', type=int, help='Override neural-network epochs (not a full default reproduction)')
    parser.add_argument('--steps', type=int, default=350)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or (args.epochs is not None and args.epochs <= 0):
        parser.error('epochs and steps must be positive')
    destination = args.output or ROOT / 'results' / ('reimplementation-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    kinds = ['himmelblau', 'toy', 'tiny-mnist', 'mnist'] if args.experiment == 'all' else [args.experiment]
    metadata = dict(started=datetime.now().isoformat(), python=sys.version,
                    torch=torch.__version__, torchvision=torchvision.__version__,
                    numpy=np.__version__, platform=platform.platform(), device='cpu', threads=1,
                    args=vars(args) | {'output': str(destination)}, experiments={},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT / 'reproduction').glob('*.py'))})
    save_json(destination / 'status.json', metadata)
    print(f'Output: {destination}', flush=True)
    for kind in kinds:
        folder = destination / kind
        folder.mkdir()
        metadata['experiments'][kind] = {'state': 'running'}
        save_json(destination / 'status.json', metadata)
        start = time.perf_counter()
        try:
            if kind == 'himmelblau':
                result = himmelblau(args.steps)
            elif kind == 'mnist':
                result = mnist(50 if args.epochs is None else args.epochs)
            else:
                result = small_comparison(kind, args.epochs)
            save_outputs(folder, kind, result)
        except BaseException as error:
            metadata['experiments'][kind].update(state='failed', error=str(error))
            save_json(destination / 'status.json', metadata)
            raise
        metadata['experiments'][kind].update(state='completed', seconds=time.perf_counter()-start)
        save_json(destination / 'status.json', metadata)
        print(f'{kind}: completed', flush=True)
    metadata['finished'] = datetime.now().isoformat()
    save_json(destination / 'status.json', metadata)


if __name__ == '__main__':
    main()
