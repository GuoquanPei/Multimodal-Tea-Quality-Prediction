"""Load the processed public subset without pickle or model dependencies."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'processed'


def _data_path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f'Path outside data directory: {relative}')
    return path


def verify_dataset(data_dir=DEFAULT_DATA):
    """Verify the size and SHA-256 of every published data file."""
    root = Path(data_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = _data_path(root, relative)
        if not path.is_file():
            raise ValueError(f'Missing data file: {relative}')
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'Size mismatch: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {relative}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Unexpected or unlisted file in the data directory')
    return manifest


def load_dataset(data_dir=DEFAULT_DATA):
    """Return samples with original-scale tabular data and processed image arrays.

    Keys match the local research dataset, with sample_id added for alignment.
    Standardization/feature fitting must be performed on training samples only.
    """
    root = Path(data_dir).resolve()
    manifest = verify_dataset(root)
    schema = json.loads((root / 'schema.json').read_text(encoding='utf-8'))
    with (root / 'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest['sample_count'] or not rows:
        raise ValueError('Sample count does not match manifest')
    samples, seen = [], set()
    for row in rows:
        identifier = row['sample_id']
        if not identifier or identifier in seen or row['region'] not in manifest['regions']:
            raise ValueError(f'Invalid sample or region: {identifier}')
        seen.add(identifier)
        sample = {'sample_id': identifier, 'region': row['region'],
                  'altitude': float(row['altitude'])}
        for key, prefix, count, dtype in [('soil', 'soil', 16, np.float64),
                                          ('Weather_index', 'weather', 6, np.float64),
                                          ('LCC_N', 'lcc_n', 12, np.float64),
                                          ('quality', 'quality', 30, np.float32)]:
            sample[key] = np.array([row[f'{prefix}_{i:02}'] for i in range(1, count + 1)], dtype=dtype)
            if not np.isfinite(sample[key]).all():
                raise ValueError(f'Non-finite values: {identifier}/{key}')
        if not np.isfinite(sample['altitude']):
            raise ValueError(f'Non-finite altitude: {identifier}')
        for key in ['MSS', 'front_rgb_image', 'back_rgb_image']:
            relative = row[key + '_file']
            if relative not in manifest['files']:
                raise ValueError(f'Unlisted array: {relative}')
            array = np.load(_data_path(root, relative), allow_pickle=False)
            expected = schema['arrays'][key]
            if array.shape != tuple(expected['shape']) or str(array.dtype) != expected['dtype']:
                raise ValueError(f'Unexpected array shape/dtype: {relative}')
            if not np.isfinite(array).all():
                raise ValueError(f'Non-finite array: {relative}')
            sample[key] = array
        samples.append(sample)
    return samples


def make_group_splits(samples):
    """Leave one released tea garden out; no garden crosses train/test."""
    regions = sorted({s['region'] for s in samples})
    if len(regions) < 2:
        raise ValueError('At least two gardens are required')
    return [{'held_out_region': region,
             'train_indices': [i for i, s in enumerate(samples) if s['region'] != region],
             'test_indices': [i for i, s in enumerate(samples) if s['region'] == region]}
            for region in regions]
