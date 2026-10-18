"""Local CPU/CUDA inference in the project environment, without networking."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEVICES = {'auto', 'cpu', 'cuda'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''


class LocalBackend:
    """Run inference outside the GUI process and verify its local artifacts."""

    def __init__(self, root=None, *, timeout=3600):
        self.root = Path(root).expanduser().resolve() if root is not None else ROOT
        executable = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
        self.python = self.root.parent / '.venv' / executable
        self.checkpoint = self.root / 'weights' / 'stone.pth'
        self.output_root = self.root / 'outputs'
        self.timeout = timeout

    def status(self):
        required = (
            (self.python, '项目运行环境尚未安装，请先运行根目录 setup_env.ps1。'),
            (self.checkpoint, '本机 Stone331 权重缺失，请恢复 weights/stone.pth。'),
            (self.root / 'provenance.json', '本机权重校验记录缺失，请恢复 provenance.json。'),
            (self.root / 'infer.py', '本机推理程序缺失，请恢复 infer.py。'),
        )
        for path, message in required:
            if not path.is_file():
                return {'ready': False, 'message': message, 'python': str(self.python)}
        return {'ready': True, 'message': '本机检测环境已就绪，可选择自动、CPU 或 GPU。',
                'python': str(self.python)}

    def detect(self, image, selection='auto', *, grayscale=False,
               threshold=0.5, progress=lambda message: None):
        if selection not in DEVICES:
            raise ValueError('设备须为 auto、cpu 或 cuda。')
        image = Path(image).expanduser().resolve()
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError('请选择有效的 PNG、JPG、BMP 或 TIF 图像文件。')
        if not 0 <= threshold <= 1:
            raise ValueError('分割阈值须在 0 到 1 之间。')
        state = self.status()
        if not state['ready']:
            raise RuntimeError(state['message'])

        progress('正在校验本机 Stone331 权重…')
        try:
            provenance = json.loads((self.root / 'provenance.json').read_text(encoding='utf-8'))
            expected_hash = provenance['checkpoint_sha256']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError('本机权重校验记录无效，请恢复 provenance.json。') from exc
        checkpoint_hash = _sha256(self.checkpoint)
        if checkpoint_hash != expected_hash:
            raise RuntimeError('本机 Stone331 权重校验不一致，已停止检测。')
        input_hash = _sha256(image)
        output = self.output_root / ('local_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)

        command = [str(self.python), str(self.root / 'infer.py'), '--input', str(image),
                   '--output', str(output), '--checkpoint', str(self.checkpoint),
                   '--device', selection, '--threshold', str(threshold)]
        if grayscale:
            command.append('--grayscale')
        environment = os.environ.copy()
        if selection == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        environment['PYTHONNOUSERSITE'] = '1'
        environment['PYTHONIOENCODING'] = 'utf-8'
        environment['PYTHONUNBUFFERED'] = '1'
        environment.pop('PYTHONPATH', None)
        label = {'auto': '自动选择本机设备', 'cpu': '本机 CPU', 'cuda': '本机 GPU'}[selection]
        progress(f'正在使用{label}检测；大幅图像可能需要较长时间…')
        try:
            completed = subprocess.run(command, cwd=str(self.root), env=environment,
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=self.timeout, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except subprocess.TimeoutExpired as exc:
            self._write_logs(output, exc.stdout, exc.stderr)
            raise RuntimeError(f'本机检测超过 {self.timeout:g} 秒，已停止。'
                               f'请缩小图像后重试。日志：{output}') from exc
        except OSError as exc:
            raise RuntimeError('无法启动本机推理环境，请检查根目录 .venv。') from exc
        self._write_logs(output, completed.stdout, completed.stderr)
        if completed.returncode:
            detail = _text(completed.stderr).strip() or _text(completed.stdout).strip()
            raise RuntimeError('本机检测失败：' + (detail[-1800:] or '推理程序异常退出。')
                               + f'\n日志：{output}')

        progress('正在校验本机检测结果…')
        try:
            report = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'本机检测缺少有效的校验报告，结果未确认。日志：{output}') from exc
        allowed = {'auto': {'cpu', 'cuda:0'}, 'cpu': {'cpu'}, 'cuda': {'cuda:0'}}[selection]
        if (not isinstance(report, dict) or not isinstance(report.get('device'), str)
                or report['device'] not in allowed):
            raise RuntimeError('实际运行设备与所选设备不符，结果未确认。')
        if report.get('requested_device') != selection:
            raise RuntimeError('实际设备请求与本次选择不符，结果未确认。')
        if report['device'] == 'cpu' and report.get('gpu_name') is not None:
            raise RuntimeError('CPU 检测报告包含 GPU 运行信息，结果未确认。')
        if report.get('checkpoint_sha256') != checkpoint_hash or report.get('strict_state_dict') is not True:
            raise RuntimeError('实际加载的权重与本机 Stone331 权重不一致，结果未确认。')
        images = report.get('images')
        if (not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict)
                or images[0].get('name') != image.name or images[0].get('input_sha256') != input_hash
                or _sha256(image) != input_hash):
            raise RuntimeError('本机检测图像校验不一致，结果未确认。')
        if _sha256(self.checkpoint) != checkpoint_hash:
            raise RuntimeError('检测期间本机权重发生变化，结果未确认。')
        paths = {kind: output / (image.name + '.' + kind + '.png')
                 for kind in ('overlay', 'mask', 'probability')}
        expected_outputs = [*paths.values(), output / (image.name + '.probability.npy')]
        if any(not path.is_file() or not path.stat().st_size for path in expected_outputs):
            raise RuntimeError(f'本机检测输出不完整，结果未确认。日志：{output}')
        progress('本机检测完成，结果已保存。')
        return {'directory': output, 'report': report, 'device': report['device'], **paths}

    @staticmethod
    def _write_logs(output, stdout, stderr):
        (output / 'inference.stdout.log').write_text(_text(stdout), encoding='utf-8')
        (output / 'inference.stderr.log').write_text(_text(stderr), encoding='utf-8')
