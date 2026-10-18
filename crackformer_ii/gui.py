"""Local Stone331 crack segmentation GUI; inference stays in a worker.

``--self-test`` uses an actual Tk window with explicitly simulated results.
``--detect-test --image ...`` runs the actual local model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageGrab, ImageOps, ImageTk


ROOT = Path(__file__).resolve().parent
NAVY, BLUE, BG, INK, MUTED, WHITE = "#152B46", "#246BCE", "#EDF2F8", "#20324A", "#64758B", "#FFFFFF"
DEVICE_LABELS = {"auto": "自动选择", "cpu": "本机 CPU", "cuda": "本机 GPU"}
DEVICE_HINTS = {
    "auto": "优先使用本机 NVIDIA GPU；没有可用 GPU 时使用 CPU。",
    "cpu": "使用本机 CPU 处理；大图可能需要较长时间。",
    "cuda": "使用本机 NVIDIA GPU；需要兼容的显卡及驱动。",
}
RESULT_VIEWS = {
    "裂纹叠加图": ("overlay", "红色叠加：阈值后的模型预测裂纹"),
    "二值掩码": ("mask", "白色：预测裂纹像素；黑色：预测背景"),
    "概率图": ("probability", "像素裂纹概率 · 亮度越高，预测概率越大"),
}
IMAGE_TYPES = [("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")]


class Preview(ttk.Frame):
    def __init__(self, parent, title, subtitle):
        super().__init__(parent, style="Card.TFrame", padding=(16, 12))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Label(self, text=title, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.caption = ttk.Label(self, text=subtitle, style="Muted.TLabel")
        self.caption.grid(row=1, column=0, sticky="w", pady=(3, 9))
        self.canvas = tk.Canvas(self, background="#F3F6FA", borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.source = self.photo = None
        self.empty_text = "请选择本机图像"
        self.canvas.bind("<Configure>", self._redraw)

    def show(self, path=None, *, text=None):
        self.source = self.photo = None
        if path is not None:
            with Image.open(path) as opened:
                # Preserve raw pixel geometry, matching inference for EXIF images.
                self.source = opened.convert("RGB")
        if text is not None:
            self.empty_text = text
        self._redraw()

    def _redraw(self, _event=None):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 3 or height < 3:
            return
        self.canvas.delete("all")
        if self.source is None:
            self.canvas.create_text(width / 2, height / 2 - 10, text="＋" if "选择" in self.empty_text else "◇",
                                    font=("Segoe UI", 28), fill="#B6C4D7")
            self.canvas.create_text(width / 2, height / 2 + 30, text=self.empty_text,
                                    font=("Microsoft YaHei UI", 10), fill=MUTED, width=max(80, width - 36))
            return
        fitted = ImageOps.contain(self.source, (max(1, width - 16), max(1, height - 16)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(fitted)
        self.canvas.create_image(width / 2, height / 2, image=self.photo)


class CrackGUI:
    def __init__(self, root, backend=None, *, device="auto", simulated=False, show_errors=True):
        if device not in DEVICE_LABELS:
            raise ValueError("检测设备须为 auto、cpu 或 cuda。")
        if backend is None:
            from local_backend import LocalBackend
            backend = LocalBackend()
        self.root, self.backend = root, backend
        self.selection, self.simulated, self.show_errors = device, simulated, show_errors
        self.image_path = self.result = self.last_error = None
        self.detecting = self.ready = self.closed = False
        self.exit_code = self._request_id = 0
        self._events = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crack-gui")
        self._after_ids = set()
        self._build()
        self._after(80, self._poll)
        self._check_environment()

    def _build(self):
        self.root.title("石材裂纹检测 · CrackFormer-II")
        self.root.configure(background=BG)
        self.root.geometry("1180x800")
        self.root.minsize(940, 620)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background=BG, foreground=INK)
        style.configure("Card.TFrame", background=WHITE)
        style.configure("TLabel", background=WHITE, foreground=INK)
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("TButton", background="#E8EFF8", borderwidth=0, padding=(12, 8), foreground=INK)
        style.map("TButton", background=[("active", "#D6E5F8"), ("disabled", "#F0F3F7")], foreground=[("disabled", "#9AA7B7")])
        style.configure("Primary.TButton", background=BLUE, foreground=WHITE, font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#1B58AF"), ("disabled", "#A8BDD9")], foreground=[("disabled", "#F6F8FC")])
        style.configure("TCombobox", fieldbackground=WHITE, padding=(5, 5))
        style.map("TCombobox", fieldbackground=[("readonly", WHITE)], selectbackground=[("readonly", WHITE)], selectforeground=[("readonly", INK)])
        style.configure("TCheckbutton", background=WHITE, padding=(4, 4))
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("TProgressbar", troughcolor="#DDE7F3", background=BLUE, borderwidth=0)

        header = tk.Frame(self.root, background=NAVY, height=80)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.columnconfigure(0, weight=1)
        tk.Label(header, text="石材裂纹检测", font=("Microsoft YaHei UI", 21, "bold"), foreground=WHITE,
                 background=NAVY).grid(row=0, column=0, sticky="w", padx=24, pady=(12, 0))
        tk.Label(header, text="CrackFormer-II  /  Stone331 预训练模型 · 裂纹分割", font=("Microsoft YaHei UI", 10),
                 foreground="#B7CCE6", background=NAVY).grid(row=1, column=0, sticky="w", padx=25, pady=(0, 10))
        self.mode_badge = tk.Label(header, text="界面自检 · 模拟结果" if self.simulated else "本机检测 · 结果保存在本机",
                                   background="#254260", foreground="#D6E8FF", padx=14, pady=9, font=("Microsoft YaHei UI", 10))
        self.mode_badge.grid(row=0, column=1, rowspan=2, padx=24)

        controls = ttk.Frame(self.root, style="Card.TFrame", padding=(16, 12))
        controls.grid(row=1, column=0, sticky="ew", padx=20, pady=(14, 10))
        controls.columnconfigure(1, weight=1)
        self.image_button = ttk.Button(controls, text="选择图像…", command=self.choose_image)
        self.image_button.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 14))
        self.filename = tk.StringVar(value="尚未选择图像")
        ttk.Label(controls, textvariable=self.filename, style="Section.TLabel").grid(row=0, column=1, sticky="w")
        self.file_detail = tk.StringVar(value="支持 PNG / JPG / BMP / TIF")
        self.path_label = ttk.Label(controls, textvariable=self.file_detail, style="Muted.TLabel")
        self.path_label.grid(row=1, column=1, sticky="w", pady=(4, 0))
        self.detect_button = ttk.Button(controls, text="开始检测", style="Primary.TButton", command=self.start_detection)
        self.detect_button.grid(row=0, column=2, rowspan=2, sticky="ns", padx=(12, 0))

        settings = ttk.Frame(controls, style="Card.TFrame")
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(settings, text="检测设备", style="Muted.TLabel").grid(row=0, column=0, padx=(0, 8))
        self.device_choice = tk.StringVar(value=DEVICE_LABELS[self.selection])
        self.device_combo = ttk.Combobox(settings, textvariable=self.device_choice, values=list(DEVICE_LABELS.values()), state="readonly", width=12)
        self.device_combo.grid(row=0, column=1, padx=(0, 18))
        self.device_combo.bind("<<ComboboxSelected>>", self._device_changed)
        self.grayscale = tk.BooleanVar(value=False)
        self.gray_check = ttk.Checkbutton(settings, text="转为灰度", variable=self.grayscale)
        self.gray_check.grid(row=0, column=2, padx=(0, 18))
        ttk.Label(settings, text="分割阈值", style="Muted.TLabel").grid(row=0, column=3, padx=(0, 8))
        self.threshold = tk.StringVar(value="0.50")
        self.threshold_spin = ttk.Spinbox(settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold, width=5, format="%.2f")
        self.threshold_spin.grid(row=0, column=4)
        self.execution_hint = ttk.Label(controls, text=DEVICE_HINTS[self.selection], style="Muted.TLabel")
        self.execution_hint.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        previews = ttk.Frame(self.root)
        previews.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 10))
        previews.columnconfigure(0, weight=1, uniform="previews")
        previews.columnconfigure(1, weight=1, uniform="previews")
        previews.rowconfigure(0, weight=1)
        self.original = Preview(previews, "原始图像", "本机文件 · 保持原始尺寸")
        self.original.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.result_preview = Preview(previews, "检测结果", RESULT_VIEWS["裂纹叠加图"][1])
        self.result_preview.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.result_preview.show(text="完成检测后显示结果")
        self.result_view = tk.StringVar(value="裂纹叠加图")
        self.result_combo = ttk.Combobox(self.result_preview, textvariable=self.result_view, values=list(RESULT_VIEWS), state="disabled", width=12)
        self.result_combo.grid(row=0, column=0, sticky="e")
        self.result_combo.bind("<<ComboboxSelected>>", self._show_result)

        footer = ttk.Frame(self.root, padding=(22, 0, 22, 12))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="正在检查本机环境…")
        self.status_label = ttk.Label(footer, textvariable=self.status, style="Status.TLabel", wraplength=860)
        self.status_label.grid(row=0, column=0, sticky="w")
        self.folder_button = ttk.Button(footer, text="打开结果文件夹", command=self.open_results, state="disabled")
        self.folder_button.grid(row=0, column=1, rowspan=2, padx=(12, 0))
        self.summary = tk.StringVar(value="选择图像 → 本机检测 → 查看结果")
        self.summary_label = ttk.Label(footer, textvariable=self.summary, style="Status.TLabel", wraplength=860)
        self.summary_label.grid(row=1, column=0, sticky="w", pady=(4, 3))
        self.progress = ttk.Progressbar(footer, mode="indeterminate", maximum=100)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.root.bind("<Configure>", self._resize, add=True)
        self._update_controls()

    def _resize(self, event):
        if event.widget is self.root:
            width = max(200, self.root.winfo_width() - 245)
            self.status_label.configure(wraplength=width)
            self.summary_label.configure(wraplength=width)
            self.path_label.configure(wraplength=max(200, self.root.winfo_width() - 350))

    def _after(self, ms, callback):
        if self.closed:
            return None
        token = None
        def wrapped():
            self._after_ids.discard(token)
            if not self.closed:
                callback()
        token = self.root.after(ms, wrapped)
        self._after_ids.add(token)
        return token

    def _check_environment(self):
        def work():
            try:
                self._events.put(("status_ok", self.backend.status()))
            except Exception as exc:
                self._events.put(("status_error", self._error(exc)))
        self._executor.submit(work)

    def _poll(self):
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "status_ok":
                self.ready = payload.get("ready") is True
                if self.ready:
                    self.status.set("图像已就绪，可以开始检测。" if self.image_path else "本机环境已就绪。请选择图像。")
                else:
                    self.last_error = str(payload.get("message") or "本机环境尚未就绪。")
                    self.status.set(self.last_error)
            elif kind == "status_error":
                self.last_error = payload
                self.status.set("本机环境检查失败：" + payload)
            elif kind == "progress":
                request, message = payload
                if self.detecting and request == self._request_id:
                    self.status.set(str(message))
            elif kind == "detect_ok":
                request, result = payload
                if request == self._request_id:
                    self._finish_detection(result)
            elif kind == "detect_error":
                request, error = payload
                if request == self._request_id:
                    self._detection_failed(error)
            self._update_controls()
        self._after(80, self._poll)

    def _device_changed(self, _event=None):
        device = next((key for key, label in DEVICE_LABELS.items() if label == self.device_choice.get()), self.selection)
        self.set_device(device)

    def set_device(self, device):
        if device not in DEVICE_LABELS:
            raise ValueError("检测设备须为 auto、cpu 或 cuda。")
        if self.detecting:
            self.device_choice.set(DEVICE_LABELS[self.selection])
            return False
        self.selection = device
        self.device_choice.set(DEVICE_LABELS[device])
        self.execution_hint.configure(text=DEVICE_HINTS[device])
        return True

    def _update_controls(self):
        locked = self.detecting
        for widget in (self.image_button, self.gray_check, self.threshold_spin):
            widget.configure(state="disabled" if locked else "normal")
        self.device_combo.configure(state="disabled" if locked else "readonly")
        self.detect_button.configure(state="normal" if self.ready and self.image_path is not None and not locked else "disabled", text="检测中…" if locked else "开始检测")
        self.folder_button.configure(state="normal" if self.result is not None else "disabled")
        self.result_combo.configure(state="readonly" if self.result is not None else "disabled")

    def choose_image(self):
        if self.detecting:
            return
        path = filedialog.askopenfilename(parent=self.root, title="选择待检测的石材图像", filetypes=IMAGE_TYPES)
        if path:
            self.load_image(path)

    def load_image(self, path):
        if self.detecting:
            return False
        try:
            path = Path(path).expanduser().resolve()
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                raise ValueError("请选择 PNG、JPG、BMP 或 TIF 图像。")
            with Image.open(path) as opened:
                opened.verify()
            self.original.show(path)
        except Exception as exc:
            self.last_error = self._error(exc)
            self.status.set("无法打开图像：" + self.last_error)
            if self.show_errors:
                messagebox.showerror("无法打开图像", self.last_error, parent=self.root)
            return False
        self.image_path, self.last_error = path, None
        self.filename.set(path.name if len(path.name) < 65 else path.name[:35] + "…" + path.name[-22:])
        width, height = self.original.source.size
        self.file_detail.set(f"{width:,} × {height:,} 像素  ·  {path.suffix[1:].upper()}  ·  本机文件")
        self.result = None
        self.result_view.set("裂纹叠加图")
        self.result_preview.caption.configure(text=RESULT_VIEWS["裂纹叠加图"][1])
        self.result_preview.show(text="点击「开始检测」获取本图结果")
        self.summary.set("选择图像 → 本机检测 → 查看结果")
        if self.ready:
            self.status.set("图像已就绪，可以开始检测。")
        self._update_controls()
        return True

    def start_detection(self):
        if self.detecting or self.image_path is None or not self.ready:
            return False
        try:
            threshold = float(self.threshold.get())
            if not 0 <= threshold <= 1:
                raise ValueError
        except (ValueError, tk.TclError):
            self.last_error = "分割阈值须为 0 到 1 之间的数字。"
            self.status.set(self.last_error)
            if self.show_errors:
                messagebox.showerror("阈值无效", self.last_error, parent=self.root)
            return False
        self.detecting, self.last_error, self.result = True, None, None
        self._request_id += 1
        request = self._request_id
        image, selection, grayscale = self.image_path, self.selection, bool(self.grayscale.get())
        self.result_preview.show(text="正在本机检测，请稍候…")
        self.status.set("正在启动本机检测…")
        self.summary.set(f"当前图像：{image.name}  ·  阈值 {threshold:g}  ·  {DEVICE_LABELS[selection]}")
        self.progress.start(12)
        self._update_controls()
        def work():
            try:
                result = self.backend.detect(image, selection=selection, grayscale=grayscale, threshold=threshold,
                    progress=lambda message: self._events.put(("progress", (request, message))))
                self._events.put(("detect_ok", (request, result)))
            except Exception as exc:
                self._events.put(("detect_error", (request, self._error(exc))))
        self._executor.submit(work)
        return True

    def _detection_failed(self, error):
        self.detecting, self.last_error, self.result = False, str(error), None
        self.progress.stop()
        self.status.set("检测未完成：" + self.last_error)
        self.result_preview.show(text="本次检测未完成，请检查后重试")
        if self.show_errors:
            messagebox.showerror("检测未完成", self.last_error, parent=self.root)
        self._update_controls()

    def _finish_detection(self, result):
        self.detecting = False
        self.progress.stop()
        try:
            report = result["report"]
            with Image.open(result["mask"]) as mask:
                count = sum(mask.convert("L").histogram()[1:])
                width, height = mask.size
            self.result = result
            self._show_result()
            fraction = count / max(1, width * height)
            elapsed = report.get("elapsed_seconds")
            timing = f"  ·  推理 {elapsed:.2f} 秒" if isinstance(elapsed, (float, int)) else ""
            device = str(result.get("device") or report.get("device") or "未知")
            device_label = "本机 GPU" if device.startswith("cuda") else "本机 CPU" if device == "cpu" else device
            prefix = "界面自检完成 · 模拟结果" if self.simulated else "检测完成"
            self.status.set(f"{prefix} · {device_label} · 结果已保存至本机")
            self.summary.set(f"{width:,} × {height:,} 像素  ·  预测裂纹 {count:,} 像素（{fraction:.4%}）{timing}")
        except Exception as exc:
            self._detection_failed("结果读取失败：" + self._error(exc))
        self._update_controls()

    def _show_result(self, _event=None):
        if self.result is not None:
            kind, caption = RESULT_VIEWS[self.result_view.get()]
            self.result_preview.show(self.result[kind])
            self.result_preview.caption.configure(text=caption)

    def open_results(self):
        if self.result is not None:
            try:
                os.startfile(str(Path(self.result["directory"]).resolve()))
            except OSError as exc:
                self.status.set("无法打开结果文件夹：" + self._error(exc))
                if self.show_errors:
                    messagebox.showerror("无法打开结果文件夹", self._error(exc), parent=self.root)

    @staticmethod
    def _error(exc):
        return str(exc).strip()[:1400] or "操作失败，请检查本机环境。"

    def save_screenshot(self, path):
        self.root.update_idletasks()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab(bbox=(x, y, x + self.root.winfo_width(), y + self.root.winfo_height())).save(path)
        return path

    def close(self, *, force=False):
        if self.detecting and not force:
            if not messagebox.askokcancel("检测仍在进行", "当前检测尚未结束，关闭界面后任务可能继续运行。确定关闭？", parent=self.root):
                return
        self.closed = True
        for token in list(self._after_ids):
            self.root.after_cancel(token)
        self._after_ids.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


class _SimulatedBackend:
    """Interface fixture, never evidence of model inference or CUDA availability."""
    def __init__(self, destination):
        self.destination = Path(destination)

    def status(self):
        return {"ready": True, "message": "界面自检", "python": sys.executable}

    def detect(self, image, selection="auto", *, grayscale=False, threshold=0.5, progress=lambda message: None):
        progress("界面自检：正在生成模拟结果…")
        time.sleep(0.4)
        self.destination.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as opened:
            original = opened.convert("RGB")
        if grayscale:
            original = ImageOps.grayscale(original).convert("RGB")
        mask = Image.new("L", original.size, 0)
        paths = {kind: self.destination / f"simulated.{kind}.png" for kind in ("overlay", "mask", "probability")}
        original.save(paths["overlay"])
        mask.save(paths["mask"])
        mask.save(paths["probability"])
        report = {"simulated": True, "accuracy_evaluated": False, "elapsed_seconds": 0.4,
                  "device": "cpu", "preprocessing": {"grayscale": grayscale, "threshold": threshold}}
        return {"directory": self.destination, "report": report, "device": "cpu", **paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description="石材裂纹检测本机 GUI")
    parser.add_argument("--image", type=Path, help="启动时载入本机图像")
    parser.add_argument("--device", choices=list(DEVICE_LABELS), default="auto", help="auto 自动选择 / cpu 本机 CPU / cuda 本机 GPU")
    tests = parser.add_mutually_exclusive_group()
    tests.add_argument("--self-test", action="store_true", help="真实窗口的界面自检，使用模拟结果，不运行模型")
    tests.add_argument("--detect-test", action="store_true", help="执行一次真实本机模型检测并关闭窗口")
    parser.add_argument("--screenshot", type=Path, help="测试完成后保存窗口截图与验证记录")
    args = parser.parse_args(argv)
    if args.detect_test and args.image is None:
        parser.error("--detect-test 必须指定 --image")
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    test_mode = args.self_test or args.detect_test
    destination = ROOT / "outputs" / ("gui_self_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        backend = _SimulatedBackend(destination) if args.self_test else None
        app = CrackGUI(root, backend, device=args.device, simulated=args.self_test, show_errors=not test_mode)
    except Exception as exc:
        messagebox.showerror("无法启动", str(exc), parent=root)
        root.destroy()
        return 1
    image = args.image
    if args.self_test and image is None:
        destination.mkdir(parents=True, exist_ok=True)
        image = destination / "simulated_input.png"
        fixture = Image.new("RGB", (640, 480), "#D7D1C6")
        ImageDraw.Draw(fixture).text((24, 24), "GUI TEST - NO MODEL INFERENCE", fill="#746E64")
        fixture.save(image)
    if image is not None:
        app._after(100, lambda: app.load_image(image))
    if test_mode:
        started = time.monotonic()
        checks = {"simulated": args.self_test, "real_model_test": args.detect_test, "requested_device": args.device}
        def finish():
            screenshot = args.screenshot or destination / "gui.png"
            try:
                checks["screenshot"] = str(app.save_screenshot(screenshot))
            except Exception as exc:
                checks["screenshot_error"] = str(exc)
                app.exit_code = 1
            expected = ("input_preview_loaded", "submitted", "settings_locked",
                        "device_change_rejected", "result_loaded", "settings_unlocked")
            failed = [name for name in expected if checks.get(name) is not True]
            for name in ("preview_switches", "device_switches"):
                if name in checks and not all(checks[name].values()):
                    failed.append(name)
            if failed:
                checks["failed_checks"] = failed
                app.exit_code = 1
            if app.exit_code:
                print(json.dumps(checks, ensure_ascii=False), file=sys.stderr)
            report_file = Path(screenshot).with_suffix(".json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
            app.close(force=True)
        def advance():
            if app.last_error:
                checks["error"] = app.last_error
                app.exit_code = 1
                finish()
                return
            if time.monotonic() - started > (3600 if args.detect_test else 30):
                checks["error"] = "测试超时"
                app.exit_code = 1
                finish()
                return
            if app.result is not None:
                checks["result_loaded"] = True
                checks["result_directory"] = str(app.result["directory"])
                checks["device"] = app.result.get("device") or app.result["report"].get("device")
                checks["preview_switches"] = {}
                for view in RESULT_VIEWS:
                    app.result_view.set(view)
                    app._show_result()
                    checks["preview_switches"][view] = app.result_preview.source is not None
                app.result_view.set("裂纹叠加图")
                app._show_result()
                checks["settings_unlocked"] = all(str(widget["state"]) != "disabled" for widget in
                    (app.image_button, app.gray_check, app.threshold_spin, app.device_combo))
                if args.detect_test and app.result["report"].get("simulated", False):
                    checks["error"] = "真实模型测试收到模拟结果"
                    app.exit_code = 1
                app._after(250, finish)
                return
            if not checks.get("submitted") and app.image_path and app.ready:
                checks["input_preview_loaded"] = app.original.source is not None
                checks["device_choices"] = list(app.device_combo["values"])
                if args.self_test:
                    checks["device_switches"] = {device: app.set_device(device) and app.selection == device for device in DEVICE_LABELS}
                    app.set_device(args.device)
                checks["submitted"] = app.start_detection()
                if checks["submitted"]:
                    checks["settings_locked"] = all(str(widget["state"]) == "disabled" for widget in
                        (app.image_button, app.gray_check, app.threshold_spin, app.device_combo))
                    checks["device_change_rejected"] = app.set_device("cpu" if args.device != "cpu" else "auto") is False
            app._after(80, advance)
        app._after(200, advance)
    root.mainloop()
    return app.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
