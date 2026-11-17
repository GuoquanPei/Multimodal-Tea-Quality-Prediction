"""Check the released subset, file integrity, and garden-separated splits."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from scripts.dataset import load_dataset, make_group_splits, verify_dataset

DATA = Path(__file__).resolve().parents[1] / 'data' / 'processed'
REGIONS = {'2_MH_guangjingmeng': 3, '3_MH_mangda': 5, '4_MH_8haoshuiku': 5}


class PublicDataTests(unittest.TestCase):
    def test_only_authorized_gardens_are_released(self):
        data = load_dataset(DATA)
        self.assertEqual(len(data), 13)
        self.assertEqual({r: sum(s['region'] == r for s in data) for r in REGIONS}, REGIONS)
        self.assertEqual({s['region'] for s in data}, set(REGIONS))
        self.assertEqual(len({s['sample_id'] for s in data}), 13)

    def test_processed_inputs_and_labels_are_complete(self):
        for sample in load_dataset(DATA):
            for name, shape in [('MSS', (4, 45, 110)),
                                ('front_rgb_image', (3, 224, 224)),
                                ('back_rgb_image', (3, 224, 224)),
                                ('soil', (16,)), ('Weather_index', (6,)),
                                ('LCC_N', (12,)), ('quality', (30,))]:
                self.assertEqual(sample[name].shape, shape)
                self.assertTrue(np.isfinite(sample[name]).all())
            self.assertGreater(float(sample['altitude']), 0)
            self.assertEqual(sample['MSS'].dtype, np.float32)
            self.assertGreater(sample['MSS'].max(), 1)  # original scale, no global fitting

    def test_no_unlisted_data_files(self):
        manifest = verify_dataset(DATA)
        expected = set(manifest['files']) | {'manifest.json'}
        actual = {p.relative_to(DATA).as_posix() for p in DATA.rglob('*') if p.is_file()}
        self.assertEqual(actual, expected)
        self.assertEqual(sum(p.endswith('.npy') for p in actual), 39)

    def test_corrupted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / 'processed'
            shutil.copytree(DATA, copied)
            target = next(copied.rglob('MSS.npy'))
            with target.open('ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Checksum|Size'):
                verify_dataset(copied)

    def test_folds_never_mix_the_same_garden(self):
        samples = load_dataset(DATA)
        folds = make_group_splits(samples)
        self.assertEqual(len(folds), 3)
        tested = []
        for fold in folds:
            train, test = fold['train_indices'], fold['test_indices']
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(range(13)))
            self.assertFalse({samples[i]['region'] for i in train} & {samples[i]['region'] for i in test})
            tested.extend(test)
        self.assertEqual(sorted(tested), list(range(13)))

    def test_csv_and_schema_preserve_column_order(self):
        schema = json.loads((DATA / 'schema.json').read_text(encoding='utf-8'))
        with (DATA / 'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 13)
        self.assertEqual(len(schema['quality_columns']), 30)
        self.assertEqual(schema['arrays']['MSS']['bands'], ['Green', 'NIR', 'Red', 'RedEdge'])
        self.assertEqual([name for name in rows[0] if name.startswith('quality_')],
                         [f'quality_{i:02}' for i in range(1, 31)])


if __name__ == '__main__':
    unittest.main()
