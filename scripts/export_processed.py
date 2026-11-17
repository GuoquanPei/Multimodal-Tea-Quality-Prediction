"""Author-side export of ONLY gardens 2/3/4 from the trusted processed snapshot.

The public data reader never uses pickle. This exporter requires joblib and
must only be run against the author's own trusted local pickle files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

REGIONS = ('2_MH_guangjingmeng', '3_MH_mangda', '4_MH_8haoshuiku')
ARRAYS = {'MSS': (4, 45, 110), 'front_rgb_image': (3, 224, 224),
          'back_rgb_image': (3, 224, 224)}


def write_json(path, content):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(content, ensure_ascii=False, indent=2) + '\n')


def export(source, metadata, output):
    import joblib  # optional dependency for the author-side export only
    samples = joblib.load(source)
    meta = joblib.load(metadata)
    selected = []
    for sample in samples:
        match = re.fullmatch(r'(.+?)\s*_\s*Q(\d+)', str(sample['region']))
        if match and match.group(1).strip() in REGIONS:
            selected.append((sample, match.group(1).strip(), int(match.group(2))))
    counts = {r: sum(region == r for _, region, _ in selected) for r in REGIONS}
    if counts != dict(zip(REGIONS, (3, 5, 5))):
        raise ValueError(f'Expected 13 samples from the authorized gardens, got {counts}')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be absent or empty; existing data will not be overwritten')
    if len(meta['quality_columns']) != 30 or len(meta['soil_columns']) != 16:
        raise ValueError('Unexpected source metadata')
    selected.sort(key=lambda item: (REGIONS.index(item[1]), item[2]))
    # Check all data before writing. Keep the processed snapshot values unchanged.
    for sample, region, number in selected:
        for key, shape in ARRAYS.items():
            a = np.asarray(sample[key])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f'Invalid processed array: {region}/Q{number}/{key}')
        for key, size in [('soil', 16), ('Weather_index', 6), ('LCC_N', 12), ('quality', 30), ('altitude', 1)]:
            a = np.asarray(sample[key])
            if a.size != size or not np.isfinite(a).all():
                raise ValueError(f'Invalid tabular values: {region}/Q{number}/{key}')
        if np.max(sample['MSS']) <= 1 or float(sample['altitude']) <= 0:
            raise ValueError('Use the original-scale Data_T_all.pkl snapshot, not Data_T.pkl')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, region, number in selected:
        folder = output / region / f'Q{number:02}'
        folder.mkdir(parents=True)
        row = {'sample_id': str(sample['region']), 'region': region}
        for key in ARRAYS:
            path = folder / (key + '.npy')
            np.save(path, np.asarray(sample[key]), allow_pickle=False)
            row[key + '_file'] = path.relative_to(output).as_posix()
        row['altitude'] = float(sample['altitude'])
        for key, prefix in [('soil', 'soil'), ('Weather_index', 'weather'),
                            ('LCC_N', 'lcc_n'), ('quality', 'quality')]:
            row.update({f'{prefix}_{i:02}': float(v) for i, v in enumerate(np.ravel(sample[key]), 1)})
        rows.append(row)
    with (output / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    stats = ['mean', 'sample_std', 'population_std', 'sample_variance', 'population_variance', 'median']
    schema = {
        'version': 1,
        'arrays': {key: {'shape': list(shape), 'dtype': 'float32'} for key, shape in ARRAYS.items()},
        'quality_columns': [{'column': f'quality_{i:02}', 'name': name, 'unit': None}
                            for i, name in enumerate(meta['quality_columns'], 1)],
        'soil_columns': [{'column': f'soil_{i:02}', 'name_and_unit': name.replace('\n', ' ')}
                         for i, name in enumerate(meta['soil_columns'], 1)],
        'weather_columns': dict(zip([f'weather_{i:02}' for i in range(1, 7)],
                                    ['annual_growing_degree_days', 'annual_mean_temperature',
                                     'annual_mean_daily_temperature_range', 'annual_total_precipitation',
                                     'annual_mean_dew_point_temperature', 'annual_mean_solar_radiation'])),
        'lcc_n_columns': dict(zip([f'lcc_n_{i:02}' for i in range(1, 13)],
                                  [f'{name}_{stat}' for name in ['chlorophyll', 'nitrogen'] for stat in stats])),
        'altitude_unit': 'm',
        'tabular_processing': 'Original-scale values; no full-dataset standardization or feature selection applied.',
        'unit_note': 'Quality units are not recorded in the source metadata; no units are inferred. Contact the author for assay units and protocols. Weather and LCC/N values retain the source measurement conventions.'}
    schema['arrays']['MSS'].update(bands=['Green', 'NIR', 'Red', 'RedEdge'],
                                  processing='Cropped and resized CHW snapshot; original numerical scale, no global min/max fitting.')
    for key in ['front_rgb_image', 'back_rgb_image']:
        schema['arrays'][key].update(channels=['R', 'G', 'B'],
                                    processing='Resized to 224x224; RGB / 255 then ImageNet channel normalization.',
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    write_json(output / 'schema.json', schema)
    files = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'format_version': 1, 'sample_count': len(rows), 'regions': counts,
                'source_snapshot': source.name,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'selection': 'Only the model inputs and quality labels of gardens 2, 3, and 4.',
                'files': files}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Trusted local Data_T_all.pkl')
    parser.add_argument('--metadata', required=True, type=Path, help='Trusted local Data_Meta_Info.pkl')
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    manifest = export(args.source.resolve(), args.metadata.resolve(), args.output.resolve())
    print(json.dumps({'samples': manifest['sample_count'], 'regions': manifest['regions']}, indent=2))


if __name__ == '__main__':
    main()
