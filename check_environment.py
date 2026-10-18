"""Check the local Stone331 inference runtime without loading the model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 CUDA 版 PyTorch 与项目依赖；无 GPU 时默认验证 CPU。")
    parser.add_argument("--require-cuda", action="store_true", help="必须具备可用 CUDA GPU，否则检查失败")
    parser.add_argument("--output", type=Path, help="保存 JSON 检查报告")
    parser.add_argument("--json", action="store_true", help="在终端输出 JSON")
    args = parser.parse_args()
    report = {
        "ok": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "require_cuda": args.require_cuda,
        "versions": {},
        "cuda_runtime": None,
        "cuda_built": None,
        "cuda_available": None,
        "device_count": None,
        "selected_device": None,
        "actual_test_device": None,
        "checks": {},
        "errors": [],
    }

    def check(name, operation):
        try:
            detail = operation()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            report["checks"][name] = {"ok": False, "error": message}
            report["errors"].append(f"{name}: {message}")
            report["ok"] = False
            return False
        report["checks"][name] = {"ok": True, **(detail or {})}
        return True

    def python_version():
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("项目环境需要 Python 3.10。")
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("请使用项目 .venv 内的解释器。")
        return {"version": report["python"]}
    check("Python 3.10 虚拟环境", python_version)

    modules = {}
    dependencies = {
        "torch": "torch", "torchvision": "torchvision", "timm": "timm",
        "numpy": "numpy", "cv2": "opencv-python-headless", "PIL": "Pillow",
        "tkinter": None,
    }
    for name, distribution in dependencies.items():
        def load(name=name, distribution=distribution):
            module = importlib.import_module(name)
            modules[name] = module
            installed = version(distribution) if distribution else str(module.TkVersion)
            report["versions"][name] = installed
            return {"version": installed}
        check(f"导入 {name}", load)

    if "torch" in modules:
        torch = modules["torch"]

        def core_versions():
            expected = {"torch": "2.1.2+cu118", "torchvision": "0.16.2+cu118"}
            for name, required in expected.items():
                if report["versions"].get(name) != required:
                    raise RuntimeError(f"{name} 需要 {required}，当前为 {report['versions'].get(name)}。")
            return expected
        check("PyTorch 核心版本", core_versions)

        def cuda_wheel():
            report["cuda_runtime"] = torch.version.cuda
            report["cuda_built"] = torch.backends.cuda.is_built()
            if report["cuda_runtime"] != "11.8" or not report["cuda_built"]:
                raise RuntimeError("需要编译了 CUDA 11.8 支持的 PyTorch，请检查是否误装 CPU 版或其他 CUDA 版本。")
            return {"cuda_runtime": report["cuda_runtime"], "cuda_built": True}
        check("CUDA 11.8 安装包", cuda_wheel)

        def devices():
            report["cuda_available"] = torch.cuda.is_available()
            report["device_count"] = torch.cuda.device_count()
            device = torch.device("cuda" if report["cuda_available"] else "cpu")
            report["selected_device"] = str(device)
            if args.require_cuda and not report["cuda_available"]:
                raise RuntimeError("指定了 --require-cuda，但当前没有可用 CUDA GPU；不会改用 CPU 代替 GPU 验证。")
            return {"cuda_available": report["cuda_available"],
                    "device_count": report["device_count"], "selected_device": str(device)}
        devices_ok = check("设备检查", devices)

        def cpu_and_numpy():
            if "numpy" not in modules:
                raise RuntimeError("NumPy 导入失败，无法验证张量互转。")
            np = modules["numpy"]
            source = np.array([[1, 2], [3, 4]], dtype=np.float32)
            with torch.inference_mode():
                left = torch.from_numpy(source.copy())
                right = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device="cpu")
                result = (left @ right).numpy()
                restored = torch.from_numpy(result.copy())
            expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
            if not np.array_equal(result, expected) or not np.array_equal(restored.numpy(), expected):
                raise RuntimeError("CPU 矩阵计算或 NumPy 双向转换结果不符。")
            return {"device": "cpu", "result": result.tolist(), "numpy_roundtrip": True}
        check("CPU 计算与 NumPy 互转", cpu_and_numpy)

        def nms():
            if "torchvision" not in modules:
                raise RuntimeError("torchvision 导入失败，无法验证 NMS 算子。")
            boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 6, 6]],
                                 dtype=torch.float32, device="cpu")
            scores = torch.tensor([0.9, 0.8, 0.7], device="cpu")
            with torch.inference_mode():
                kept = modules["torchvision"].ops.nms(boxes, scores, 0.5).tolist()
            if kept != [0, 2]:
                raise RuntimeError(f"NMS 结果不符：{kept}")
            return {"device": "cpu", "kept_indices": kept}
        check("torchvision NMS", nms)

        if devices_ok:
            def selected_device_test():
                device = torch.device(report["selected_device"])
                with torch.inference_mode():
                    matrix = torch.ones((2, 2), device=device)
                    result = matrix @ matrix
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    actual = str(result.device)
                    values = result.cpu().tolist()
                report["actual_test_device"] = actual
                if values != [[2.0, 2.0], [2.0, 2.0]]:
                    raise RuntimeError(f"设备矩阵计算结果不符：{values}")
                detail = {"device": actual, "result": values}
                if device.type == "cuda":
                    detail["gpu_name"] = torch.cuda.get_device_name(device)
                return detail
            check("所选设备实际计算", selected_device_test)

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            report["ok"] = False
            report["errors"].append(f"保存报告失败: {exc}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("环境检查：" + ("通过" if report["ok"] else "失败"))
        print(f"Python {report['python']}：{report['executable']}")
        print(f"CUDA 运行库：{report['cuda_runtime']}；编译支持：{report['cuda_built']}")
        print(f"GPU 可用：{report['cuda_available']}；设备数量：{report['device_count']}")
        print(f"实际计算设备：{report['actual_test_device']}")
        for name, result in report["checks"].items():
            detail = result.get("version") or result.get("error") or ""
            print(f"[{'通过' if result['ok'] else '失败'}] {name}" + (f"：{detail}" if detail else ""))
        if args.output and report["ok"]:
            print(f"报告：{args.output.resolve()}")
        for error in report["errors"]:
            print(f"错误：{error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
