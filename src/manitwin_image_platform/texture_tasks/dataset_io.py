"""任务文件读写及可解码的配对续跑。"""
import json
import os
from pathlib import Path

from PIL import Image

from .planner import TextureTaskGroup
from .renderer import render_texture_pair


def group_root(root: Path, split: str, group_id: str) -> Path:
    """定位与现有ABFolder一致的任务目录。"""
    return root / split / 'texture_relation' / group_id


def decodable_png(path: Path, size: int | None = None) -> bool:
    """解码实际PNG像素，识别大于128字节的截断文件。"""
    try:
        with Image.open(path) as image:
            if image.format != 'PNG' or (size is not None and image.size != (size, size)):
                return False
            image.load()
            return image.mode == 'RGB'
    except (OSError, ValueError, SyntaxError):
        return False


def atomic_save(image: Image.Image, path: Path) -> None:
    """完整写好PNG后再替换目标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.png')
    image.save(temporary, format='PNG', optimize=False)
    os.replace(temporary, path)


def group_complete(root: Path, group: TextureTaskGroup, image_size: int) -> bool:
    """只跳过原型、槽位和实际像素均完整的任务。"""
    folder = group_root(root, group.split, group.group_id)
    try:
        payload = json.loads((folder / 'group.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    expected = group.to_dict()
    for key in ('schema_version', 'task_id', 'prototype_a', 'prototype_b', 'scenes', 'recipe'):
        if payload.get(key) != expected[key]:
            return False
    count = len(group.scenes)
    if len(payload.get('slots', [])) != count:
        return False
    for role in ('A', 'B'):
        files = list((folder / role).glob('*.png'))
        if len(files) != count:
            return False
        if not all(decodable_png(folder / role / f'{i:03d}.png', image_size) for i in range(count)):
            return False
    return True


def render_group(root: Path, group: TextureTaskGroup, image_size: int) -> tuple[str, bool]:
    """续跑以整对为单位修复，保持最终颜色直方图匹配。"""
    folder = group_root(root, group.split, group.group_id)
    if group_complete(root, group, image_size):
        return group.group_id, False
    slots = []
    for index, scene in enumerate(group.scenes):
        image_a, image_b, metrics = render_texture_pair(group, scene, image_size, image_size)
        if metrics['foreground_rgb_mae'] < 1.0:
            raise ValueError(f'{group.task_id}/slot{index}: 最终图像类别差异不可见')
        for role, image in (('A', image_a), ('B', image_b)):
            atomic_save(image, folder / role / f'{index:03d}.png')
        slots.append({
            'slot_index': index, 'scene_id': scene.scene_id,
            'a_path': f'A/{index:03d}.png', 'b_path': f'B/{index:03d}.png',
            'shape_family': scene.shape_family, 'shape_domain': scene.shape_domain,
            'background_style': scene.background_style, 'pixel_metrics': metrics,
        })
    payload = group.to_dict()
    payload.update(slots=slots, images_per_class=len(group.scenes), declared_stage='generated')
    (folder / 'group.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return group.group_id, True


def write_group_summary(root: Path, plan: list[TextureTaskGroup], filename: str) -> None:
    """导出训练读取器可直接定位的任务清单。"""
    with (root / filename).open('w', encoding='utf-8') as handle:
        for group in plan:
            folder = group_root(root, group.split, group.group_id)
            row = {
                'schema_version': 4, 'task_family': 'texture_relation',
                'task_id': group.task_id, 'task_kind': group.task_kind,
                'decision_factor': group.decision_factor, 'recipe': group.recipe,
                'layout_domain': group.layout_domain, 'layout_combination': group.layout_combination,
                'group_id': group.group_id, 'split': group.split,
                'hard_pair': group.hard_pair, 'texture_domain': group.texture_domain,
                'family_a': group.prototype_a.family, 'family_b': group.prototype_b.family,
                'group_root': str(folder), 'a_dir': str(folder / 'A'), 'b_dir': str(folder / 'B'),
                'images_a': len(group.scenes), 'images_b': len(group.scenes),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
