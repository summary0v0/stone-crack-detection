"""Inference against the unmodified official CrackFormer-II network."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_device(selection, torch_module):
    """Select a local device once; a requested CUDA device never falls back."""
    if selection not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('设备须为 auto、cpu 或 cuda。')
    if selection == 'cpu':
        return 'cpu'
    available = torch_module.cuda.is_available()
    if selection == 'cuda' and not available:
        raise RuntimeError('本机 CUDA GPU 不可用。请检查 NVIDIA 驱动及 CUDA 版 PyTorch；未改用 CPU。')
    return 'cuda:0' if available else 'cpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='One image or a directory of images')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'prediction')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'weights' / 'stone.pth')
    parser.add_argument('--device', '--gpu', dest='device', choices=['auto', 'cpu', 'cuda'],
                        default='auto', help='Local device: auto, cpu, or cuda (no fallback when cuda is requested)')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--color', choices=['bgr', 'rgb'], default='bgr')
    parser.add_argument('--normalization', choices=['unit', 'symmetric'], default='unit')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Check checkpoint and a synthetic 512x512 forward pass')
    args = parser.parse_args()
    if args.tile_size < 32 or args.tile_size % 32 or not 0 <= args.overlap < args.tile_size:
        parser.error('tile-size must be a positive multiple of 32; overlap must be smaller than tile-size')
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    if not args.smoke and args.input is None:
        parser.error('--input or --smoke is required')
    import cv2
    import numpy as np
    import torch
    import timm
    import torchvision
    try:
        device = resolve_device(args.device, torch)
    except RuntimeError as exc:
        raise SystemExit('CUDA_UNAVAILABLE: ' + str(exc)) from None
    sys.path.insert(0, str(ROOT / 'upstream' / 'CrackFormer-II'))
    from nets.crackformerII import crackformer

    torch.set_num_threads(4)
    checkpoint_hash = sha256(args.checkpoint)
    if args.checkpoint.resolve() == (ROOT / 'weights' / 'stone.pth').resolve():
        provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8'))
        if checkpoint_hash != provenance['checkpoint_sha256']:
            raise ValueError('Stone331 权重 SHA-256 校验不一致，已停止检测。')
    model = crackformer()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected a state_dict checkpoint')
    if checkpoint and all(k.startswith('module.') for k in checkpoint):
        checkpoint = {k[7:]: value for k, value in checkpoint.items()}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint tensor: {key}')
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'timm': timm.__version__,
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'cuda_runtime': torch.version.cuda, 'requested_device': args.device, 'device': device,
        'gpu_name': torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
        'checkpoint_sha256': checkpoint_hash, 'strict_state_dict': True,
        'parameters': sum(p.numel() for p in model.parameters()),
        'checkpoint_keys': len(checkpoint), 'input_channels': 3,
        'preprocessing': {'color': args.color, 'normalization': args.normalization,
                          'grayscale': args.grayscale, 'tile_size': args.tile_size,
                          'overlap': args.overlap, 'threshold': args.threshold},
        'accuracy_evaluated': False,
    }

    def forward(image):
        if args.color == 'rgb':
            image = image[:, :, ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        if args.normalization == 'symmetric':
            tensor = (tensor - 0.5) / 0.5
        with torch.inference_mode():
            outputs = model(tensor.unsqueeze(0).to(device))
            if len(outputs) != 6 or outputs[-1].shape != (1, 1, args.tile_size, args.tile_size):
                raise RuntimeError('Unexpected model output shape')
            probability = outputs[-1].sigmoid()[0, 0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise RuntimeError('Non-finite output')
        return probability

    started = time.perf_counter()
    if args.smoke:
        probability = forward(np.full((args.tile_size, args.tile_size, 3), 127, np.uint8))
        report['smoke'] = {'shape': list(probability.shape), 'finite': True,
                           'minimum': float(probability.min()), 'maximum': float(probability.max()),
                           'purpose': 'deployment functionality only; synthetic image is not an accuracy test'}
    else:
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        inputs = [args.input] if args.input.is_file() else sorted(
            p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        if not inputs:
            raise ValueError('No supported images found')
        # Prevent output collisions such as foo.jpg and foo.png.
        if len({p.name for p in inputs}) != len(inputs):
            raise ValueError('Duplicate image names')
        report['images'] = []
        for path in inputs:
            input_bytes = path.read_bytes()
            input_hash = hashlib.sha256(input_bytes).hexdigest()
            image = cv2.imdecode(np.frombuffer(input_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8:
                raise ValueError(f'Expected an 8-bit image; convert 16-bit/raw data with an explicit scaling policy: {path.name}')
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if args.grayscale:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            h, w = image.shape[:2]
            padded = cv2.copyMakeBorder(image, 0, max(0, args.tile_size - h),
                                        0, max(0, args.tile_size - w), cv2.BORDER_REPLICATE)
            ph, pw = padded.shape[:2]
            def starts(length):
                return sorted(set(list(range(0, length - args.tile_size + 1,
                                              args.tile_size - args.overlap)) + [length - args.tile_size]))
            total = np.zeros((ph, pw), np.float32)
            count = np.zeros((ph, pw), np.float32)
            tiles = 0
            for y in starts(ph):
                for x in starts(pw):
                    patch = forward(padded[y:y + args.tile_size, x:x + args.tile_size])
                    total[y:y + args.tile_size, x:x + args.tile_size] += patch
                    count[y:y + args.tile_size, x:x + args.tile_size] += 1
                    tiles += 1
            probability = (total / count)[:h, :w]
            mask = (probability >= args.threshold).astype(np.uint8) * 255
            overlay = image.copy()
            selected = mask.astype(bool)
            overlay[selected] = (0.5 * overlay[selected] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            prefix = args.output / path.name
            np.save(str(prefix) + '.probability.npy', probability)
            for suffix, pixels in [('probability.png', np.rint(probability * 255).astype(np.uint8)),
                                    ('mask.png', mask), ('overlay.png', overlay)]:
                encoded, buffer = cv2.imencode('.png', pixels)
                if not encoded:
                    raise RuntimeError('PNG encoding failed')
                Path(str(prefix) + '.' + suffix).write_bytes(buffer.tobytes())
            if sha256(path) != input_hash:
                raise RuntimeError(f'检测期间输入图像发生变化：{path.name}')
            report['images'].append({'name': path.name, 'input_sha256': input_hash,
                                      'shape': [h, w], 'tiles': tiles,
                                      'crack_pixel_fraction': float(selected.mean()),
                                      'probability_min': float(probability.min()),
                                      'probability_max': float(probability.max())})
    report['elapsed_seconds'] = time.perf_counter() - started
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('检测期间模型权重发生变化，结果未确认。')
    if device.startswith('cuda'):
        report['peak_cuda_memory_mib'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    report_path = args.output / 'verification.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
