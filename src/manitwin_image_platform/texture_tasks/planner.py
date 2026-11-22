"""以任务为单位平衡纹理与布局关系，并保留独立验证原型。"""
from dataclasses import dataclass
import json
import random
from pathlib import Path

from .layout import LAYOUT_FAMILIES, LayoutPrototype, make_layout_pair
from .prototype import (
    HELDOUT_TEXTURE_FAMILIES, TRAIN_TEXTURE_FAMILIES, TexturePrototype,
    make_distinct_same_family_pair, prototype_distance, sample_texture_prototype,
)
from .scene import SceneSpec, build_group_scenes

RECIPE_COUNTS = {'texture': (3, 1, 0), 'layout': (1, 1, 2)}
Prototype = TexturePrototype | LayoutPrototype


@dataclass(frozen=True, slots=True)
class TextureTaskGroup:
    """描述一个完整的临时A/B任务。"""
    group_id: str
    split: str
    group_index: int
    hard_pair: bool
    texture_domain: str
    prototype_a: Prototype
    prototype_b: Prototype
    scenes: tuple[SceneSpec, ...]
    task_kind: str = 'texture_cross'
    decision_factor: str = 'texture_family'
    recipe: str = 'texture'
    layout_domain: str = 'not_applicable'
    layout_combination: str = ''

    @property
    def task_id(self) -> str:
        """让训练和验证任务ID在整个数据集中唯一。"""
        return f'texture_relation/{self.split}/{self.group_id}'

    def to_dict(self) -> dict:
        """导出训练读取器可独立消费的任务信息。"""
        return {
            'schema_version': 4, 'task_id': self.task_id, 'group_id': self.group_id,
            'split': self.split, 'group_index': self.group_index,
            'hard_pair': self.hard_pair, 'texture_domain': self.texture_domain,
            'task_kind': self.task_kind, 'decision_factor': self.decision_factor,
            'recipe': self.recipe, 'declared_stage': 'planned',
            'layout_domain': self.layout_domain, 'layout_combination': self.layout_combination,
            'prototype_a': self.prototype_a.to_dict(), 'prototype_b': self.prototype_b.to_dict(),
            'prototype_distance': prototype_distance(self.prototype_a, self.prototype_b),
            'scenes': [scene.to_dict() for scene in self.scenes],
        }


def _task_kinds(count: int, recipe: str | None, hard_ratio: float) -> list[str]:
    """固定T/L配方；直接调用旧接口时仍支持自定义困难组比例。"""
    if recipe is None:
        return ['texture_hard' if round((i + 1) * hard_ratio) > round(i * hard_ratio)
                else 'texture_cross' for i in range(count)]
    cross, hard, layout = RECIPE_COUNTS[recipe]
    cycle = ['texture_cross'] * cross + ['texture_hard'] * hard + ['layout'] * layout
    return [cycle[i % len(cycle)] for i in range(count)]


def _texture_pair(group_id: str, family: str, pool: tuple[str, ...], hard: bool,
                  seeds: tuple[int, int], counts: dict[str, int], rotation: int) -> tuple[TexturePrototype, TexturePrototype]:
    """让跨家族配对覆盖不同组合，困难对使用有效参数。"""
    if hard:
        return make_distinct_same_family_pair(group_id, family, *seeds)
    candidates = [f for f in pool if f != family]
    minimum = min(counts.get(f, 0) for f in candidates)
    candidates = [f for f in candidates if counts.get(f, 0) == minimum]
    other = candidates[rotation % len(candidates)]
    return (sample_texture_prototype(f'{group_id}_A_proto', family, seeds[0]),
            sample_texture_prototype(f'{group_id}_B_proto', other, seeds[1]))


def build_group_plan(split: str, group_count: int, images_per_class: int, seed: int, *,
                     hard_pair_ratio: float = 0.25, heldout_texture_ratio: float = 0.25,
                     heldout_shape_ratio: float = 0.25, recipe: str | None = None) -> list[TextureTaskGroup]:
    """生成确定性的分组计划；验证原型和形状独立留出。"""
    if split not in {'train', 'val'} or group_count < 1 or images_per_class < 8:
        raise ValueError('split必须为train/val，组数大于0，每类至少8张')
    if recipe is not None and recipe not in RECIPE_COUNTS:
        raise ValueError('recipe必须是texture或layout')
    if any(not 0 <= value <= 1 for value in (hard_pair_ratio, heldout_texture_ratio, heldout_shape_ratio)):
        raise ValueError('困难组和留出比例必须在[0,1]')
    rng = random.Random(seed + (0 if split == 'train' else 900_001))
    kinds = _task_kinds(group_count, recipe, hard_pair_ratio)
    occurrence: dict[tuple[str, str], int] = {}
    totals: dict[str, int] = {}
    role_a: dict[str, int] = {}
    role_b: dict[str, int] = {}
    plan = []
    texture_index = layout_index = 0
    for idx, kind in enumerate(kinds):
        group_id = f'group_{idx:06d}'
        seeds = (rng.randint(1, 2_000_000_000), rng.randint(1, 2_000_000_000))
        domain, hard = 'seen', kind == 'texture_hard'
        layout_domain, combination = 'not_applicable', ''
        swap = (idx // 8) % 2
        if kind == 'layout':
            family = LAYOUT_FAMILIES[layout_index % len(LAYOUT_FAMILIES)]
            heldout_layout = split == 'val' and (layout_index // 4) % 4 == layout_index % 4
            a, b = make_layout_pair(group_id, family, seeds[0], heldout=heldout_layout)
            layout_domain = 'heldout' if heldout_layout else 'seen'
            combination = a.combination_id
            swap = (layout_index // 4 + layout_index // 16) % 2
            layout_index += 1
            factor = family
        else:
            heldout = split == 'val' and round((texture_index + 1) * heldout_texture_ratio) > round(texture_index * heldout_texture_ratio)
            domain = 'heldout' if heldout else 'seen'
            pool = HELDOUT_TEXTURE_FAMILIES if heldout else TRAIN_TEXTURE_FAMILIES
            key = (kind, domain)
            number = occurrence.get(key, 0)
            family = pool[number % len(pool)]
            occurrence[key] = number + 1
            a, b = _texture_pair(group_id, family, pool, hard, seeds, totals, idx)
            totals[a.family] = totals.get(a.family, 0) + 1
            totals[b.family] = totals.get(b.family, 0) + 1
            texture_index += 1
            factor = 'texture_structure' if hard else 'texture_family'
            forward = role_a.get(a.family, 0) + role_b.get(b.family, 0)
            reverse = role_a.get(b.family, 0) + role_b.get(a.family, 0)
            if forward != reverse:
                swap = forward > reverse
        if swap:
            a, b = b, a
        role_a[a.family] = role_a.get(a.family, 0) + 1
        role_b[b.family] = role_b.get(b.family, 0) + 1
        scenes = tuple(build_group_scenes(
            split, idx, images_per_class, seed,
            heldout_shape_ratio=0.0 if split == 'train' else heldout_shape_ratio,
        ))
        plan.append(TextureTaskGroup(group_id, split, idx, hard, domain, a, b, scenes,
                                     kind, factor, recipe or 'texture', layout_domain, combination))
    return plan


def save_group_plan(plan: list[TextureTaskGroup], path: str | Path) -> None:
    """保存冻结计划，后续续跑读取同一份原型与场景。"""
    with Path(path).open('w', encoding='utf-8') as handle:
        for group in plan:
            handle.write(json.dumps(group.to_dict(), ensure_ascii=False) + '\n')


def _proto(record: dict) -> Prototype:
    """根据原型种类还原纹理或布局对象。"""
    values = dict(record)
    kind = values.pop('prototype_kind', 'texture')
    return LayoutPrototype(**values) if kind == 'layout' else TexturePrototype(**values)


def load_group_plan(path: str | Path) -> list[TextureTaskGroup]:
    """读取现有分组计划并补齐历史纹理任务的说明字段。"""
    groups = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hard = bool(row['hard_pair'])
            groups.append(TextureTaskGroup(
                row['group_id'], row['split'], int(row['group_index']), hard,
                row['texture_domain'], _proto(row['prototype_a']), _proto(row['prototype_b']),
                tuple(SceneSpec(**scene) for scene in row['scenes']),
                row.get('task_kind', 'texture_hard' if hard else 'texture_cross'),
                row.get('decision_factor', 'texture_structure' if hard else 'texture_family'),
                row.get('recipe', 'texture'),
                row.get('layout_domain', 'not_applicable'), row.get('layout_combination', ''),
            ))
    return groups
