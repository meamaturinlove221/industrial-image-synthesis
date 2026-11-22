"""后台生成与深度审计，所有界面更新经消息队列返回。"""
from __future__ import annotations

import json
import subprocess
import sys
import threading


def start_job(function, *args):
    """启动单个后台任务，避免阻塞 Tk 主线程。"""
    threading.Thread(target=function, args=args, daemon=True).start()


def generation_job(project_root, options, messages):
    """调用正式生成入口并保留完整输出日志及失败原因。"""
    python = project_root / ".venv/Scripts/python.exe"
    command = [str(python) if python.is_file() else sys.executable, str(project_root / "tools/texture_task_generate.py")]
    for key, value in options.items():
        command.extend((f"--{key}", str(value)))
    logs = project_root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "texture_task_generation.log"
    tail = []
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    tail = (tail + [line.rstrip()])[-12:]
                    messages.put(("line", line.rstrip()))
                code = process.wait()
        if code:
            raise RuntimeError(f"生成退出码 {code}\n日志：{log_path}\n" + "\n".join(tail))
        messages.put(("done", "数据生成完成且图像检查通过；训练与真实回归请查看独立阶段结果。"))
    except Exception as exc:
        messages.put(("error", str(exc)))


def audit_job(dataset_root, messages):
    """逐图解码并运行任务审计，原始图像不作修改。"""
    from manitwin_image_platform.texture_tasks.audit import audit_texture_task_dataset
    try:
        report = audit_texture_task_dataset(dataset_root, deep_image_check=True)
        if dataset_root.is_dir():
            (dataset_root / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not report.get("passed"):
            raise RuntimeError("\n".join(report.get("errors", ["审计未通过"])[:12]))
        messages.put(("done", "图像解码与任务审计通过；该结果不代表模型识别已经达标。"))
    except Exception as exc:
        messages.put(("error", str(exc)))
