"""Exercise local device selection without allocating CUDA memory."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    'local_infer_device', Path(__file__).resolve().parents[1] / 'infer.py')
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


class DeviceTests(unittest.TestCase):
    def runtime(self, available):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=available)))

    def test_auto_uses_cpu_without_cuda(self):
        runtime = self.runtime(False)
        self.assertEqual(infer.resolve_device('auto', runtime), 'cpu')

    def test_auto_uses_local_cuda_when_available(self):
        runtime = self.runtime(True)
        self.assertEqual(infer.resolve_device('auto', runtime), 'cuda:0')

    def test_cpu_selection_does_not_query_or_allocate_cuda(self):
        runtime = self.runtime(True)
        self.assertEqual(infer.resolve_device('cpu', runtime), 'cpu')
        runtime.cuda.is_available.assert_not_called()

    def test_cuda_selection_cannot_fall_back(self):
        runtime = self.runtime(False)
        with self.assertRaisesRegex(RuntimeError, '未改用 CPU'):
            infer.resolve_device('cuda', runtime)
        runtime.cuda.is_available.assert_called_once_with()

    def test_cuda_selection_uses_local_cuda(self):
        self.assertEqual(infer.resolve_device('cuda', self.runtime(True)), 'cuda:0')

    def test_old_physical_gpu_selector_is_rejected(self):
        with self.assertRaises(ValueError):
            infer.resolve_device('GPU-server-id', self.runtime(True))

    def test_module_import_needs_no_remote_or_torch_runtime(self):
        isolated = importlib.util.module_from_spec(SPEC)
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split('.')[0],
                             {'remote', 'paramiko', 'dotenv', 'gpu_policy', 'fcntl', 'torch'})
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded_import), \
             patch.object(Path, 'read_text', side_effect=AssertionError('unexpected configuration read')):
            SPEC.loader.exec_module(isolated)


if __name__ == '__main__':
    unittest.main()
