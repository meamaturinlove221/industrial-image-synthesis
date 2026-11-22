"""核对分组语义、独立留出和最终可解码像素，不把数据审计当模型效果。"""
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from .dataset_io import group_root
from .planner import load_group_plan
from .prototype import HELDOUT_TEXTURE_FAMILIES, prototype_distance
from .renderer import pair_pixel_metrics, render_foreground_mask
from .shape_bank import HELDOUT_SHAPE_FAMILIES


def _read_group(folder: Path, expected: dict, errors: list[str]) -> dict:
    """核对每组真实JSON，防止计划与训练读取信息不一致。"""
    try:
        payload = json.loads((folder / 'group.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        errors.append(f'{folder}: group.json不可读取：{exc}')
        return {}
    for key in ('group_id', 'split', 'prototype_a', 'prototype_b', 'scenes'):
        if payload.get(key) != expected[key]:
            errors.append(f'{folder.name}: {key}与冻结计划不同')
    if payload.get('schema_version') == 4:
        for key in ('task_id', 'task_kind', 'decision_factor', 'recipe', 'layout_domain', 'layout_combination'):
            if payload.get(key) != expected[key]:
                errors.append(f'{folder.name}: {key}与冻结计划不同')
    return payload


def _pair_check(folder: Path, index: int, scene, image_size: int | None, errors: list[str]) -> dict:
    """实际解码A/B并检查可见差异、共享背景及前景直方图。"""
    paths = [folder / role / f'{index:03d}.png' for role in ('A', 'B')]
    try:
        images = []
        for path in paths:
            with Image.open(path) as source:
                source.load()
                if source.format != 'PNG' or source.mode != 'RGB':
                    raise ValueError(f'必须为RGB PNG：{path}')
                if image_size is not None and source.size != (image_size, image_size):
                    raise ValueError(f'图像尺寸与声明不同：{path}')
                images.append(source.copy())
        if images[0].size != images[1].size:
            raise ValueError('A/B图像尺寸不同')
        mask = render_foreground_mask(scene, *images[0].size)
        metrics = pair_pixel_metrics(*images, mask)
        if not metrics['background_equal']:
            errors.append(f'{folder.name}/slot{index}: A/B实际背景不同')
        if metrics['foreground_histogram_l1'] != 0:
            errors.append(f'{folder.name}/slot{index}: 可见前景颜色直方图不匹配')
        if metrics['foreground_rgb_mae'] < 1.0:
            errors.append(f'{folder.name}/slot{index}: 最终像素类别差异不可见')
        return metrics
    except (OSError, ValueError, SyntaxError) as exc:
        errors.append(f'{folder.name}/slot{index}: 图片不可解码或不符合声明：{exc}')
        return {}


def audit_texture_task_dataset(root: str | Path, *, deep_image_check: bool = True) -> dict:
    """审计冻结计划、实际文件与可见像素，清楚标明是否执行深检。"""
    root = Path(root).resolve()
    errors: list[str] = []
    report = {
        'schema_version': 4, 'root': str(root), 'passed': False,
        'deep_image_check': deep_image_check, 'errors': errors, 'warnings': [],
        'declared_stage': 'audit_failed',
    }
    try:
        plans = {split: load_group_plan(root / f'{split}_plan.jsonl') for split in ('train', 'val')}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f'无法读取训练/验证计划：{exc}')
        return report
    summary_path = root / 'dataset_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    size = summary.get('image_size')
    train_seeds = {p.seed for g in plans['train'] for p in (g.prototype_a, g.prototype_b)}
    val_seeds = {p.seed for g in plans['val'] for p in (g.prototype_a, g.prototype_b)}
    overlap = train_seeds & val_seeds
    if overlap:
        errors.append(f'训练/验证原型种子重叠：{len(overlap)}')
    if any(p.family in HELDOUT_TEXTURE_FAMILIES for g in plans['train'] for p in (g.prototype_a, g.prototype_b)):
        errors.append('训练集使用了未见纹理家族')
    if any(s.shape_family in HELDOUT_SHAPE_FAMILIES for g in plans['train'] for s in g.scenes):
        errors.append('训练集使用了未见形状家族')
    train_combinations = {g.layout_combination for g in plans['train'] if g.task_kind == 'layout'}
    heldout_combinations = {g.layout_combination for g in plans['val'] if g.layout_domain == 'heldout'}
    if train_combinations & heldout_combinations or any(g.layout_domain == 'heldout' for g in plans['train']):
        errors.append('训练集使用了验证保留的布局组合')
    ids, duplicate_ids, inspected_pairs = set(), 0, 0
    split_reports = {}
    for split, plan in plans.items():
        domain_counts, shape_domains = Counter(), Counter()
        kind_counts, factors, backgrounds = Counter(), Counter(), Counter()
        expected_ids = {g.group_id for g in plan}
        split_root = root / split / 'texture_relation'
        present_ids = {p.name for p in split_root.iterdir() if p.is_dir()} if split_root.is_dir() else set()
        if present_ids != expected_ids:
            errors.append(f'{split}: 实际任务目录与计划不同')
        for group in plan:
            duplicate_ids += int(group.task_id in ids)
            ids.add(group.task_id)
            domain_counts[group.texture_domain] += 1
            kind_counts[group.task_kind] += 1
            factors[group.decision_factor] += 1
            if group.hard_pair and prototype_distance(group.prototype_a, group.prototype_b) < 0.22:
                errors.append(f'{group.task_id}: 有效参数差异不足')
            folder = group_root(root, split, group.group_id)
            payload = _read_group(folder, group.to_dict(), errors)
            slots = payload.get('slots', [])
            if len(slots) != len(group.scenes):
                errors.append(f'{group.task_id}: 槽位数量不一致')
            for role in ('A', 'B'):
                names = {p.name for p in (folder / role).glob('*.png')}
                if names != {f'{i:03d}.png' for i in range(len(group.scenes))}:
                    errors.append(f'{group.task_id}: {role}图片清单不完整')
            for index, scene in enumerate(group.scenes):
                shape_domains[scene.shape_domain] += 1
                backgrounds[scene.background_style] += 1
                if index < len(slots):
                    slot = slots[index]
                    expected = {'slot_index': index, 'scene_id': scene.scene_id,
                                'a_path': f'A/{index:03d}.png', 'b_path': f'B/{index:03d}.png',
                                'shape_domain': scene.shape_domain}
                    if any(slot.get(key) != value for key, value in expected.items()):
                        errors.append(f'{group.task_id}/slot{index}: 路径或场景信息不一致')
                if deep_image_check:
                    metrics = _pair_check(folder, index, scene, size, errors)
                    inspected_pairs += int(bool(metrics))
            if len(group.scenes) <= 8:
                if len({s.shape_family for s in group.scenes}) != len(group.scenes):
                    errors.append(f'{group.task_id}: 形状覆盖不足')
                if len({s.background_style for s in group.scenes}) != len(group.scenes):
                    errors.append(f'{group.task_id}: 背景覆盖不足')
        if summary.get('task_kind_counts', {}).get(split, dict(kind_counts)) != dict(kind_counts):
            errors.append(f'{split}: 配方计数与摘要不同')
        split_reports[split] = {
            'planned_groups': len(plan), 'present_groups': len(present_ids),
            'task_kind_counts': dict(kind_counts), 'decision_factor_counts': dict(factors),
            'texture_domain_counts': dict(domain_counts), 'shape_domain_slot_counts': dict(shape_domains),
            'background_style_slot_counts': dict(backgrounds),
            'layout_domain_counts': dict(Counter(g.layout_domain for g in plan if g.task_kind == 'layout')),
        }
    if duplicate_ids:
        errors.append(f'跨分割任务ID重复：{duplicate_ids}')
    report.update(
        passed=not errors, recipe=summary.get('recipe'), train_groups=len(plans['train']),
        val_groups=len(plans['val']), splits=split_reports, inspected_pairs=inspected_pairs,
        train_val_prototype_seed_overlap=len(overlap), duplicate_prototype_pairs=duplicate_ids,
        val_heldout_texture_groups=sum(g.texture_domain == 'heldout' for g in plans['val']),
        val_heldout_shape_slots=sum(s.shape_domain == 'heldout' for g in plans['val'] for s in g.scenes),
        val_heldout_layout_groups=sum(g.layout_domain == 'heldout' for g in plans['val']),
        declared_stage='audited' if not errors and deep_image_check else 'structure_checked' if not errors else 'audit_failed',
    )
    return report
