"""Check released data integrity and display its contents."""
import argparse
from collections import Counter
import json

from dataset import DEFAULT_DATA, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DEFAULT_DATA)
    args = parser.parse_args()
    samples = load_dataset(args.data_dir)
    print(json.dumps({'samples': len(samples),
                      'regions': dict(Counter(s['region'] for s in samples)),
                      'MSS_shape': list(samples[0]['MSS'].shape),
                      'RGB_shape': list(samples[0]['front_rgb_image'].shape),
                      'quality_targets': len(samples[0]['quality']),
                      'checksums': 'verified'}, indent=2))


if __name__ == '__main__':
    main()
