"""Verify local subprocess routing and artifact guards without a model runtime."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_backend


class LocalBackendTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.folder.name) / 'crackformer_ii'
        self.root.mkdir()
        self.backend = local_backend.LocalBackend(self.root)
        self.backend.python.parent.mkdir(parents=True)
        self.backend.python.write_bytes(b'interpreter fixture')
        self.backend.checkpoint.parent.mkdir()
        self.backend.checkpoint.write_bytes(b'checkpoint fixture')
        self.checkpoint_hash = hashlib.sha256(self.backend.checkpoint.read_bytes()).hexdigest()
        (self.root / 'provenance.json').write_text(json.dumps(
            {'checkpoint_sha256': self.checkpoint_hash}), encoding='utf-8')
        (self.root / 'infer.py').write_text('# inference fixture', encoding='utf-8')
        self.image = self.root / '完整石板图.png'
        self.image.write_bytes(b'input fixture')
        self.result_device = 'cpu'
        self.mutate_report = lambda report: None
        self.mutate_outputs = lambda destination: None

    def tearDown(self):
        self.folder.cleanup()

    def fake_run(self, command, **kwargs):
        self.command, self.process_options = command, kwargs
        destination = Path(command[command.index('--output') + 1])
        self.report = {
            'device': self.result_device,
            'requested_device': command[command.index('--device') + 1],
            'checkpoint_sha256': self.checkpoint_hash, 'strict_state_dict': True,
            'images': [{'name': self.image.name,
                        'input_sha256': hashlib.sha256(self.image.read_bytes()).hexdigest()}],
        }
        self.mutate_report(self.report)
        (destination / 'verification.json').write_text(json.dumps(self.report), encoding='utf-8')
        for suffix in ('overlay.png', 'mask.png', 'probability.png', 'probability.npy'):
            (destination / (self.image.name + '.' + suffix)).write_bytes(b'output fixture')
        self.mutate_outputs(destination)
        return subprocess.CompletedProcess(command, 0, 'successful run', '')

    def detect(self, **kwargs):
        with patch.object(local_backend.subprocess, 'run', side_effect=self.fake_run):
            return self.backend.detect(self.image, **kwargs)

    def test_cpu_uses_root_environment_and_isolated_child(self):
        result = self.detect(selection='cpu', grayscale=True, threshold=0.35)
        self.assertEqual(self.backend.python.parent.parent, self.root.parent / '.venv')
        self.assertEqual(self.command[0], str(self.backend.python))
        self.assertEqual(self.command[self.command.index('--device') + 1], 'cpu')
        self.assertEqual(self.command[self.command.index('--checkpoint') + 1], str(self.backend.checkpoint))
        self.assertIn('--grayscale', self.command)
        self.assertEqual(self.command[self.command.index('--threshold') + 1], '0.35')
        self.assertNotIn('shell', self.process_options)
        self.assertEqual(self.process_options['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(self.process_options['env']['PYTHONNOUSERSITE'], '1')
        self.assertNotIn('PYTHONPATH', self.process_options['env'])
        self.assertEqual(self.process_options['timeout'], 3600)
        if os.name == 'nt':
            self.assertEqual(self.process_options['creationflags'], subprocess.CREATE_NO_WINDOW)
        self.assertEqual(result['device'], 'cpu')
        self.assertEqual(set(result), {'directory', 'report', 'device', 'overlay', 'mask', 'probability'})
        self.assertEqual((result['directory'] / 'inference.stdout.log').read_text(encoding='utf-8'),
                         'successful run')

    def test_auto_accepts_both_local_devices_and_keeps_cuda_visibility(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            for device in ('cpu', 'cuda:0'):
                with self.subTest(device=device):
                    self.result_device = device
                    self.assertEqual(self.detect()['device'], device)
                    self.assertEqual(self.process_options['env']['CUDA_VISIBLE_DEVICES'], '0')

    def test_relative_root_is_resolved_before_child_cwd_changes(self):
        relative = os.path.relpath(self.root, Path.cwd())
        self.backend = local_backend.LocalBackend(relative)
        self.detect(selection='cpu')
        self.assertEqual(self.process_options['cwd'], str(self.root.resolve()))
        self.assertTrue(Path(self.command[0]).is_absolute())
        self.assertTrue(Path(self.command[1]).is_absolute())

    def test_explicit_cuda_accepts_cuda_report(self):
        self.result_device = 'cuda:0'
        self.assertEqual(self.detect(selection='cuda')['device'], 'cuda:0')

    def test_explicit_device_mismatch_is_rejected(self):
        for selection, actual in (('cpu', 'cuda:0'), ('cuda', 'cpu'), ('auto', 'cuda:1')):
            with self.subTest(selection=selection, actual=actual):
                self.result_device = actual
                with self.assertRaisesRegex(RuntimeError, '设备不符'):
                    self.detect(selection=selection)

    def test_malformed_device_report_is_rejected(self):
        self.mutate_report = lambda report: report.update(device=['cpu'])
        with self.assertRaisesRegex(RuntimeError, '设备不符'):
            self.detect()

    def test_report_must_match_requested_device(self):
        self.mutate_report = lambda report: report.update(requested_device='cpu')
        with self.assertRaisesRegex(RuntimeError, '设备请求'):
            self.detect(selection='auto')

    def test_status_and_import_do_not_read_configuration_or_import_remote(self):
        spec = importlib.util.spec_from_file_location('isolated_local_backend', local_backend.__file__)
        isolated = importlib.util.module_from_spec(spec)
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split('.')[0], {'remote', 'paramiko', 'dotenv', 'torch', 'gpu_policy'})
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded_import), \
             patch.object(Path, 'read_text', side_effect=AssertionError('unexpected configuration read')):
            spec.loader.exec_module(isolated)
            self.assertTrue(isolated.LocalBackend(self.root).status()['ready'])

    def test_missing_runtime_does_not_launch(self):
        self.backend.python.unlink()
        with patch.object(local_backend.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, '环境尚未安装'):
                self.backend.detect(self.image)
            run.assert_not_called()

    def test_invalid_selection_does_not_launch(self):
        with patch.object(local_backend.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                self.backend.detect(self.image, 'remote')
            run.assert_not_called()

    def test_corrupt_weight_does_not_launch(self):
        self.backend.checkpoint.write_bytes(b'changed checkpoint')
        with patch.object(local_backend.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, '权重校验不一致'):
                self.backend.detect(self.image)
            run.assert_not_called()

    def test_changed_input_during_inference_is_rejected(self):
        self.mutate_outputs = lambda destination: self.image.write_bytes(b'changed image')
        with self.assertRaisesRegex(RuntimeError, '图像校验不一致'):
            self.detect()

    def test_changed_weight_during_inference_is_rejected(self):
        self.mutate_outputs = lambda destination: self.backend.checkpoint.write_bytes(b'changed weight')
        with self.assertRaisesRegex(RuntimeError, '权重发生变化'):
            self.detect()

    def test_wrong_weight_report_is_rejected(self):
        self.mutate_report = lambda report: report.update(checkpoint_sha256='incorrect')
        with self.assertRaisesRegex(RuntimeError, '实际加载的权重'):
            self.detect()

    def test_non_strict_weight_report_is_rejected(self):
        self.mutate_report = lambda report: report.update(strict_state_dict=False)
        with self.assertRaisesRegex(RuntimeError, '实际加载的权重'):
            self.detect()

    def test_incomplete_artifacts_are_rejected(self):
        self.mutate_outputs = lambda destination: (destination / (self.image.name + '.probability.npy')).unlink()
        with self.assertRaisesRegex(RuntimeError, '输出不完整'):
            self.detect()

    def test_missing_report_is_rejected(self):
        self.mutate_outputs = lambda destination: (destination / 'verification.json').unlink()
        with self.assertRaisesRegex(RuntimeError, '有效的校验报告'):
            self.detect()

    def test_failed_cuda_process_surfaces_error_without_retry(self):
        with patch.object(local_backend.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 1, '', 'CUDA_UNAVAILABLE')) as run:
            with self.assertRaisesRegex(RuntimeError, 'CUDA_UNAVAILABLE'):
                self.backend.detect(self.image, 'cuda')
            self.assertEqual(run.call_count, 1)

    def test_timeout_is_stopped_without_fallback(self):
        with patch.object(local_backend.subprocess, 'run', side_effect=
                          subprocess.TimeoutExpired('python', 3600, output=b'partial output')) as run:
            with self.assertRaisesRegex(RuntimeError, '超过 3600 秒'):
                self.backend.detect(self.image)
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
