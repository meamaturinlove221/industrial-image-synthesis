"""主窗口布局：参数、任务表、成对预览及实验阶段。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from manitwin_image_platform.texture_tasks.prototype import (
    HELDOUT_TEXTURE_FAMILIES, TEXTURE_FAMILY_CN, TRAIN_TEXTURE_FAMILIES,
)


def entry_row(parent, row, title, variable, command=None):
    """创建带可选目录按钮的一行输入。"""
    ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", pady=3)
    ttk.Entry(parent, textvariable=variable, width=26).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
    if command:
        ttk.Button(parent, text="选择", command=command).grid(row=row, column=2)


def build_controls(app, parent):
    """构建固定配方及数据规模配置，显示实际任务组成。"""
    panel = ttk.LabelFrame(parent, text="一、数据规模与配方", padding=10)
    panel.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
    panel.columnconfigure(1, weight=1)
    entry_row(panel, 0, "输出根目录", app.output_root, app._pick_output)
    entry_row(panel, 1, "数据集名称", app.dataset_name)
    ttk.Label(panel, text="任务配方").grid(row=2, column=0, sticky="w")
    ttk.Combobox(panel, textvariable=app.recipe, values=("纹理与布局（L）", "纹理对照（T）"), state="readonly", width=24).grid(row=2, column=1, sticky="ew", padx=6)
    for row, title, variable in (
        (3, "训练任务组数", app.train_groups), (4, "验证任务组数", app.val_groups),
        (5, "每组每类图片", app.images_per_class), (6, "图像尺寸", app.image_size),
        (7, "随机种子", app.seed), (8, "并行线程", app.workers),
        (9, "验证未见纹理比例", app.val_heldout_texture_ratio),
        (10, "验证未见形状比例", app.val_heldout_shape_ratio),
    ):
        entry_row(panel, row, title, variable)
    ttk.Separator(panel).grid(row=11, column=0, columnspan=3, sticky="ew", pady=8)
    app.total_label = ttk.Label(panel, font=("Microsoft YaHei UI", 10, "bold"))
    app.total_label.grid(row=12, column=0, columnspan=3, sticky="w")
    descriptions = (
        "L 配方：25% 不同纹理 + 25% 同家族困难 + 50% 表面布局。\nT 对照：75% 不同纹理 + 25% 同家族困难。",
        "布局依据：区域位置、排列、分散程度和连接关系。\n不绑定真实物体标签或原图模板。",
        "训练纹理（20 种）：\n" + "、".join(TEXTURE_FAMILY_CN[x] for x in TRAIN_TEXTURE_FAMILIES),
        "验证未见纹理：" + "、".join(TEXTURE_FAMILY_CN[x] for x in HELDOUT_TEXTURE_FAMILIES),
        "每组 A/B 共用形状、背景、配色、姿态与光照；每组重新定义类别。支持和查询必须来自同一个新任务，并按配对槽位分开。",
    )
    for row, text in enumerate(descriptions, 13):
        ttk.Label(panel, text=text, wraplength=350, justify="left").grid(row=row, column=0, columnspan=3, sticky="w", pady=(9, 0))
    actions = ttk.Frame(panel)
    actions.grid(row=18, column=0, columnspan=3, sticky="ew", pady=(12, 3))
    ttk.Button(actions, text="刷新任务计划", command=app._refresh_plan).pack(side="left")
    ttk.Button(actions, text="生成当前组预览", command=app._refresh_preview).pack(side="left", padx=4)
    generate = ttk.Button(actions, text="生成正式任务集", command=app._start_generation)
    generate.pack(side="left")
    actions2 = ttk.Frame(panel)
    actions2.grid(row=19, column=0, columnspan=3, sticky="ew", pady=3)
    audit = ttk.Button(actions2, text="审计现有任务集", command=app._audit)
    audit.pack(side="left")
    ttk.Button(actions2, text="打开输出根目录", command=app._open_output).pack(side="left", padx=4)
    ttk.Button(actions2, text="刷新阶段结果", command=app._refresh_stage).pack(side="left")
    ttk.Button(panel, text="进入三维任务平台", command=app._open_3d).grid(row=20, column=0, columnspan=3, sticky="ew", pady=8)
    app.job_buttons = [generate, audit]


def build_task_table(app, parent):
    """显示任务来源与判别因素，允许预览验证集的未见形状。"""
    panel = ttk.LabelFrame(parent, text="二、任务计划（前 200 组）", padding=8)
    panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
    panel.columnconfigure(0, weight=1)
    panel.rowconfigure(1, weight=1)
    toggle = ttk.Frame(panel)
    toggle.grid(row=0, column=0, sticky="w", pady=(0, 6))
    for label, value in (("训练", "train"), ("验证", "val")):
        ttk.Radiobutton(toggle, text=label, value=value, variable=app.preview_split, command=app._refresh_plan).pack(side="left")
    columns = ("group", "A", "B", "factor", "domain")
    app.tree = ttk.Treeview(panel, columns=columns, show="headings")
    for name, title, width in zip(columns, ("组号", "A 原型", "B 原型", "判别依据", "纹理域"), (100, 100, 100, 120, 65)):
        app.tree.heading(name, text=title)
        app.tree.column(name, width=width, anchor="center")
    app.tree.grid(row=1, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(panel, orient="vertical", command=app.tree.yview)
    scrollbar.grid(row=1, column=1, sticky="ns")
    app.tree.configure(yscrollcommand=scrollbar.set)
    app.tree.bind("<<TreeviewSelect>>", lambda event: app._refresh_preview())


def build_previews(app, parent):
    """展示配对场景、同类变化及支持查询的槽位解释。"""
    panel = ttk.Frame(parent)
    panel.grid(row=0, column=2, sticky="nsew")
    panel.columnconfigure(0, weight=1)
    panel.rowconfigure(0, weight=1)
    panel.rowconfigure(1, weight=1)
    pair = ttk.LabelFrame(panel, text="三、严格 A/B 同场景对照", padding=6)
    pair.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
    ttk.Label(pair, text="每个槽位共用场景，只改变本任务的纹理或表面布局。").pack(anchor="w")
    app.preview_pair = ttk.Label(pair)
    app.preview_pair.pack(anchor="n", pady=5)
    same = ttk.LabelFrame(panel, text="四、同一类跨形状/背景/颜色", padding=6)
    same.grid(row=1, column=0, sticky="nsew")
    toggle = ttk.Frame(same)
    toggle.pack(fill="x")
    ttk.Label(toggle, text="结构原型保持不变，场景独立变化。").pack(side="left")
    for role in ("A", "B"):
        ttk.Radiobutton(toggle, text=f"看{role}类", variable=app.preview_role, value=role, command=app._refresh_preview).pack(side="right")
    app.preview_class = ttk.Label(same)
    app.preview_class.pack(anchor="n", pady=5)
    app.summary = tk.Text(same, width=65, height=10, wrap="word", state="disabled")
    app.summary.pack(fill="both", expand=True)


def build_window(app):
    """组织三个主要操作区和不混淆生成与识别的阶段栏。"""
    header = ttk.Frame(app, padding=(12, 10))
    header.pack(fill="x")
    ttk.Label(header, text="ManiTwin 纹理与布局任务生成", font=("Microsoft YaHei UI", 20, "bold")).pack(side="left")
    ttk.Label(header, text="C1 结构保持不变 · 同任务支持/查询 · 通用能力回归").pack(side="right")
    body = ttk.Frame(app, padding=(12, 0, 12, 8))
    body.pack(fill="both", expand=True)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)
    body.columnconfigure(2, weight=1)
    build_controls(app, body)
    build_task_table(app, body)
    build_previews(app, body)
    footer = ttk.Frame(app, padding=(12, 4, 12, 10))
    footer.pack(fill="x")
    ttk.Label(footer, textvariable=app.stage_status, wraplength=1600).pack(anchor="w", pady=4)
    ttk.Progressbar(footer, variable=app.progress, maximum=100).pack(fill="x")
    ttk.Label(footer, textvariable=app.status).pack(anchor="w", pady=(5, 0))
