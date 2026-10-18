import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_MBHT import validate_c3comb_benchmark_manifest
from scripts.build_timestamp_benchmark import build_retailrocket


class TimestampBenchmarkBuilderTest(unittest.TestCase):
    def test_every_target_occurrence_uses_its_complete_prefix(self):
        rows = [
            'timestamp,visitorid,event,itemid,transactionid',
            '0,10,view,1,',
            '10,10,view,2,',
            '20,10,view,3,',
            '30,10,transaction,3,1',
            '80,10,transaction,3,2',
            '100,10,transaction,3,3',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'events.csv'
            output = root / 'benchmark'
            source.write_text('\n'.join(rows) + '\n', encoding='utf-8')
            args = argparse.Namespace(
                source='retailrocket',
                input=source,
                output_dir=output,
                dataset_name='tiny_time_beh',
                max_history=0,
                min_history=2,
                min_item_train_events=1,
            )
            manifest = build_retailrocket(args)

            self.assertEqual(sum(manifest['split_counts'].values()), 3)
            self.assertIsNone(manifest['max_history'])
            self.assertEqual(
                manifest['history_policy'], 'complete observable prefix'
            )
            self.assertEqual(
                manifest['target_supervision_policy'],
                'every eligible target-behavior occurrence',
            )

            prefix_lengths = []
            for split in ('train', 'valid', 'development', 'confirmation'):
                path = output / 'tiny_time_beh.{}.inter'.format(split)
                for line in path.read_text(encoding='utf-8').splitlines()[1:]:
                    prefix_lengths.append(len(line.split('\t')[1].split()))
            self.assertEqual(sorted(prefix_lengths), [3, 4, 5])

            validated = validate_c3comb_benchmark_manifest(
                output, 'tiny_time_beh'
            )
            self.assertEqual(validated, manifest)

    def test_superseded_capped_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / 'old_time_beh.manifest.json'
            manifest_path.write_text(
                json.dumps({
                    'history_policy': 'most recent 24 events',
                    'max_history': 24,
                    'max_train_targets_per_user': 20,
                }),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'superseded protocol'):
                validate_c3comb_benchmark_manifest(root, 'old_time_beh')


if __name__ == '__main__':
    unittest.main()
