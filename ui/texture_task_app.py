"""纹理与布局任务主窗口：连接配置、预览及后台生成。"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from manitwin_image_platform.texture_tasks.planner import build_group_plan
from manitwin_image_platform.texture_tasks.preview import (
    group_summary_text, paired_group_sheet, same_class_across_scenes_sheet,
)
from ui.texture_task_jobs import audit_job, generation_job, start_job
from ui.texture_task_layout import build_window
from ui.texture_task_status import DECISION_LABELS, dataset_stage_text, preview_partition_text

RECIPE_LABELS = {"纹理与布局（L）": "layout", "纹理对照（T）": "texture"}


class TextureTaskApp(tk.Tk):
    """管理主窗口状态，并通过消息队列更新后台任务进度。"""

    def __init__(self, smoke=False):
        """初始化固定实验默认参数和可见操作区。"""
        super().__init__()
        self.smoke = smoke
        if smoke:
            self.withdraw()
        self.title("ManiTwin 纹理与布局任务生成 · C1 首期")
        self.geometry("1740x1000")
        self.minsize(1480, 860)
        self.output_root = tk.StringVar(value="output/texture_tasks")
        self.dataset_name = tk.StringVar(value="layout")
        self.recipe = tk.StringVar(value="纹理与布局（L）")
        self.preview_split = tk.StringVar(value="train")
        self.train_groups = tk.IntVar(value=3200)
        self.val_groups = tk.IntVar(value=640)
        self.images_per_class = tk.IntVar(value=8)
        self.image_size = tk.IntVar(value=224)
        self.seed = tk.IntVar(value=42)
        self.workers = tk.IntVar(value=4)
        self.val_heldout_texture_ratio = tk.DoubleVar(value=0.25)
        self.val_heldout_shape_ratio = tk.DoubleVar(value=0.25)
        self.preview_role = tk.StringVar(value="A")
        self.progress = tk.DoubleVar(value=0)
        self.status = tk.StringVar(value="就绪：选择任务配方，检查 A/B 与支持/查询划分。")
        self.stage_status = tk.StringVar(value="数据未生成；训练接入未验证；真实回归未验证")
        self.plan = []
        self.q = queue.Queue()
        self.busy = False
        self.active_root = None
        self.photo_pair = self.photo_class = None
        build_window(self)
        self._refresh_plan()
        self.after(120, self._drain)

    def _pick_output(self):
        """选择数据输出根目录。"""
        directory = filedialog.askdirectory(title="选择输出根目录")
        if directory:
            self.output_root.set(directory)
            self._refresh_stage()

    def _dataset_root(self):
        """取得当前配置对应的数据集目录。"""
        return Path(self.output_root.get().strip()) / self.dataset_name.get().strip()

    def _selected_group(self):
        """根据树控件选择取得原始任务对象。"""
        selection = self.tree.selection()
        return next((group for group in self.plan if group.group_id in selection), self.plan[0] if self.plan else None)

    def _refresh_plan(self):
        """生成完整规模的确定性计划，显示其前两百组。"""
        try:
            split = self.preview_split.get()
            count = self.train_groups.get() if split == "train" else self.val_groups.get()
            self.plan = build_group_plan(
                split, count, self.images_per_class.get(), self.seed.get() + (split == "val"),
                hard_pair_ratio=0.25, recipe=RECIPE_LABELS[self.recipe.get()],
                heldout_texture_ratio=self.val_heldout_texture_ratio.get() if split == "val" else 0.0,
                heldout_shape_ratio=self.val_heldout_shape_ratio.get() if split == "val" else 0.0,
            )[:200]
            self.tree.delete(*self.tree.get_children())
            for group in self.plan:
                self.tree.insert("", "end", iid=group.group_id, values=(
                    group.group_id, group.prototype_a.family_cn, group.prototype_b.family_cn,
                    DECISION_LABELS.get(group.decision_factor, group.decision_factor),
                    "未见纹理" if group.texture_domain == "heldout" else "已见纹理",
                ))
            if self.plan:
                self.tree.selection_set(self.plan[0].group_id)
            scale = self.images_per_class.get() * 2
            self.total_label.configure(text=f"训练 {self.train_groups.get() * scale:,} 张 · 验证 {self.val_groups.get() * scale:,} 张")
            self._refresh_stage()
            self._refresh_preview()
        except Exception as exc:
            if self.smoke:
                raise
            messagebox.showerror("参数错误", str(exc))

    def _refresh_preview(self):
        """显示配对渲染、类内变化和明确的槽位分法。"""
        group = self._selected_group()
        if group is None:
            return
        try:
            size = self.image_size.get()
            pair = paired_group_sheet(group, size, size, thumb=118)
            same_class = same_class_across_scenes_sheet(group, self.preview_role.get(), size, size, columns=4, thumb=118)
            pair.thumbnail((650, 330), Image.Resampling.LANCZOS)
            same_class.thumbnail((650, 280), Image.Resampling.LANCZOS)
            self.photo_pair, self.photo_class = ImageTk.PhotoImage(pair), ImageTk.PhotoImage(same_class)
            self.preview_pair.configure(image=self.photo_pair)
            self.preview_class.configure(image=self.photo_class)
            self.summary.configure(state="normal")
            self.summary.delete("1.0", "end")
            self.summary.insert("1.0", group_summary_text(group) + "\n" + preview_partition_text(group, self._dataset_root()))
            self.summary.configure(state="disabled")
            self.status.set(f"预览 {group.split}/{group.group_id} · A/B 只在本任务内定义")
        except Exception as exc:
            if self.smoke:
                raise
            messagebox.showerror("预览失败", str(exc))

    def _start_generation(self):
        """提交生成任务，保留冻结配置的续跑语义。"""
        if self.busy:
            return
        try:
            options = {
                "output-root": self.output_root.get().strip(), "dataset-name": self.dataset_name.get().strip(),
                "recipe": RECIPE_LABELS[self.recipe.get()], "train-groups": self.train_groups.get(),
                "val-groups": self.val_groups.get(), "images-per-class": self.images_per_class.get(),
                "image-size": self.image_size.get(), "seed": self.seed.get(), "workers": self.workers.get(),
                "val-heldout-texture-ratio": self.val_heldout_texture_ratio.get(),
                "val-heldout-shape-ratio": self.val_heldout_shape_ratio.get(),
            }
            self.active_root = self._dataset_root()
            self._set_busy(True)
            start_job(generation_job, ROOT, options, self.q)
        except Exception as exc:
            self._set_busy(False)
            messagebox.showerror("参数错误", str(exc))

    def _audit(self):
        """在后台解码审计，避免大数据集阻塞界面。"""
        if not self.busy:
            self.active_root = self._dataset_root()
            self._set_busy(True)
            start_job(audit_job, self.active_root, self.q)

    def _open_output(self):
        """打开当前数据集目录或其已存在的父目录。"""
        import os
        directory = self._dataset_root()
        directory = directory if directory.is_dir() else directory.parent
        try:
            os.startfile(directory)
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))

    def _open_3d(self):
        """打开独立三维任务入口，保留当前二维原型工作区。"""
        import subprocess
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen([sys.executable, str(ROOT / "ui/task3d_app.py")], cwd=ROOT, creationflags=flags)

    def _set_busy(self, busy):
        """阻止重复提交同一目录的生成或审计任务。"""
        self.busy = busy
        for button in self.job_buttons:
            button.configure(state="disabled" if busy else "normal")

    def _refresh_stage(self):
        """只根据磁盘中的实际结果显示完成阶段。"""
        self.stage_status.set(dataset_stage_text(self._dataset_root()))

    def _drain(self):
        """消费后台消息并在主线程更新控件。"""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line" and payload.startswith("@@TEXTURE_TASK_PROGRESS@@|"):
                    _, done, total, stage = payload.split("|", 3)
                    self.progress.set(100 * int(done) / max(1, int(total)))
                    self.status.set(f"{done}/{total} 组 · {stage}")
                elif kind == "done":
                    self._set_busy(False)
                    self._refresh_stage()
                    self.status.set(payload)
                elif kind == "error":
                    self._set_busy(False)
                    self._refresh_stage()
                    self.status.set("任务失败，数据与失败记录已保留")
                    messagebox.showerror("任务失败", str(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain)


def main():
    """启动窗口，或执行创建、布局和销毁的界面小测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    app = TextureTaskApp(smoke=args.smoke)
    if args.smoke:
        app.withdraw()
        app.update_idletasks()
        app.destroy()
        print("UI_SMOKE_OK")
    else:
        app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
