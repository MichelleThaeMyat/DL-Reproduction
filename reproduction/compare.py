"""Compare full small-task histories and MNIST summaries/printed epoch histories."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np


def read_csv(path):
    with path.open() as stream:
        return {row['Optimizer']: row for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Captured small-experiment folder')
    parser.add_argument('--original-mnist', type=Path, required=True, help='Original run folder containing CSV and run.log')
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-6)
    args = parser.parse_args()
    checks = []

    def check(label, expected, actual, atol=None, rtol=None):
        a, b = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
        valid = a.shape == b.shape and bool(np.isfinite(a).all() and np.isfinite(b).all())
        equal = valid and bool(np.allclose(a, b, atol=args.atol if atol is None else atol,
                                           rtol=args.rtol if rtol is None else rtol))
        checks.append(dict(check=label, passed=equal, exact=valid and bool(np.array_equal(a, b)),
                           max_absolute_difference=float(np.max(np.abs(a-b))) if valid and a.size else None))

    for kind in ['himmelblau', 'toy', 'tiny-mnist']:
        expected = json.loads((args.reference / kind / 'metrics.json').read_text())
        actual = json.loads((args.new / kind / 'metrics.json').read_text())
        if expected.keys() != actual.keys():
            raise ValueError(f'Optimizer labels differ: {kind}')
        for name, metrics in expected.items():
            for key, values in metrics.items():
                check(f'{kind}/{name}/{key}', values, actual[name][key])
    expected = read_csv(args.original_mnist / 'mnist_optimizer_comparison_summary_R6.csv')
    actual = read_csv(args.new / 'mnist' / 'summary.csv')
    if expected.keys() != actual.keys():
        raise ValueError('MNIST optimizer labels differ')
    for name, row in expected.items():
        for key in row:
            if key != 'Optimizer':
                check(f'mnist/{name}/{key}', [float(row[key])], [float(actual[name][key])])

    # Original logs store four decimal places for losses and two for accuracies.
    # Compare at precisely that precision; do not pretend raw traces were saved.
    histories = json.loads((args.new / 'mnist' / 'metrics.json').read_text())
    pattern = re.compile(r'Epoch (\d+)/(\d+) - Train Loss: ([\d.]+) - Train Acc: ([\d.]+)% - Test Loss: ([\d.]+) - Test Acc: ([\d.]+)%')
    name, counts = None, {}
    for line in (args.original_mnist / 'run.log').read_text().splitlines():
        if line.startswith('Running '):
            name = line[len('Running '):].removesuffix('...')
        match = pattern.fullmatch(line)
        if match:
            epoch, total = map(int, match.group(1, 2))
            metrics = histories[name]
            if len(metrics['train_loss']) != total:
                raise ValueError('Epoch counts differ; use full default runs for comparison')
            counts[name] = counts.get(name, 0) + 1
            values = [metrics[key][epoch-1] for key in ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']]
            rounded = [float(f'{value:.{places}f}') for value, places in zip(values, [4, 2, 4, 2])]
            check(f'mnist/{name}/epoch-{epoch}-printed-metrics',
                  list(map(float, match.group(3, 4, 5, 6))), rounded, atol=0, rtol=0)
    if counts != {name: len(data['train_loss']) for name, data in histories.items()}:
        raise ValueError('Missing or duplicate epoch records in reference log')
    report = dict(passed=all(item['passed'] for item in checks),
                  tolerance={'atol': args.atol, 'rtol': args.rtol},
                  note='MNIST summaries use full precision; epoch logs use original printed precision. '
                       'Original batch losses and corrected Himmelblau trajectories are not compared.',
                  checks=checks)
    target = args.new / 'comparison.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    failures = [item for item in checks if not item['passed']]
    print(f'{len(checks)-len(failures)}/{len(checks)} checks passed; report: {target}')
    for item in failures:
        print(item)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
