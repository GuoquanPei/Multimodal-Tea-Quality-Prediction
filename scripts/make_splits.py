"""Create three demonstration folds, holding out one entire garden per fold."""
import argparse
import json
from pathlib import Path

from dataset import DEFAULT_DATA, load_dataset, make_group_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/garden_splits.json'))
    args = parser.parse_args()
    samples = load_dataset(args.data_dir)
    content = {'method': 'leave-one-tea-garden-out',
               'note': 'Demonstration splits for the public subset, not the paper splits.',
               'sample_ids': [s['sample_id'] for s in samples],
               'folds': make_group_splits(samples)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(content["folds"])} folds to {args.output}')


if __name__ == '__main__':
    main()
