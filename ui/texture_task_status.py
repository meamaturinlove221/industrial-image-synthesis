"""读取实际数据和实验结果，展示清晰的阶段与槽位信息。"""
from __future__ import annotations

import json
from pathlib import Path

DECISION_LABELS = {
    "texture_family": "纹理规律", "texture_structure": "纹理有效属性",
    "region_position": "区域位置", "region_arrangement": "区域排列",
    "region_dispersion": "分散程度", "region_connectivity": "连接关系",
}


def read_report(path):
    """读取可选结果文件，尚未生成或正在写入时返回空结果。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def dataset_stage_text(root):
    """仅在对应证据文件通过时标记该阶段成功。"""
    state = read_report(root / "build_state.json")
    audit = read_report(root / "dataset_audit.json")
    integration = read_report(root / "training_integration.json")
    regression = read_report(root / "regression_result.json")
    generated = "数据已生成" if state.get("status") == "COMPLETED" else "数据未完成"
    checked = "图像检查通过" if audit.get("passed") and audit.get("deep_image_check") else "图像检查待验证"
    connected = "训练接入通过" if integration.get("passed") else "训练接入未验证"
    accepted = "真实回归达标" if regression.get("accepted") else "真实回归未达标" if regression else "真实回归未验证"
    return " ｜ ".join((generated, checked, connected, accepted))


def preview_partition_text(group, root):
    """优先展示实际接入记录，否则明确标注为合法槽位示例。"""
    report = read_report(root / "training_integration.json")
    record = next((item for item in report.get("episodes", []) if item.get("task_id") == getattr(group, "task_id", "")), None)
    if record:
        support, query = record["support_slots"], record["query_slots"]
        title = "已验证训练上下文"
    else:
        seen = [index for index, scene in enumerate(group.scenes) if scene.shape_domain == "seen"]
        support = seen[:4]
        query = [index for index in range(len(group.scenes)) if index not in support]
        title = "合法划分示例（实际训练四个上下文轮换）"
    return f"{title}\nSupport A/B 同槽：{support}\nQuery A/B 同槽：{query}\n支持与查询槽不重叠；验证支持只取训练已见形状。"
