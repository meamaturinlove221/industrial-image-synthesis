"""不依赖物品标签的低频位置、排列、分散和连接任务。"""
from dataclasses import asdict, dataclass
import math
import random

import numpy as np

LAYOUT_FAMILIES = ('region_position', 'region_arrangement', 'region_dispersion', 'region_connectivity')
LAYOUT_FAMILY_CN = {
    'region_coverage': '区域面积占比',
    'region_position': '区域位置', 'region_arrangement': '区域排列',
    'region_dispersion': '聚集与分散', 'region_connectivity': '连接与分离',
}


@dataclass(frozen=True, slots=True)
class LayoutPrototype:
    """保存程序化布局状态；A/B只是当前任务中的角色。"""
    prototype_id: str
    family: str
    family_cn: str
    seed: int
    variant: int
    axis_angle: float
    region_fraction: float
    element_count: int = 1
    combination_id: str = 'seen'

    def to_dict(self) -> dict:
        """导出可由纯JSON读取的原型。"""
        return {**asdict(self), 'prototype_kind': 'layout'}


def make_layout_pair(group_id: str, family: str, seed: int, *, heldout: bool = False) -> tuple[LayoutPrototype, LayoutPrototype]:
    """共享布局骨架与面积，只改变一个关系状态。"""
    rng = random.Random(seed)
    angle = rng.uniform(-0.22, 0.22)
    fraction = rng.uniform(0.20, 0.24)
    counts = {'region_coverage': (1, 2), 'region_position': (1, 2), 'region_arrangement': (3, 4),
              'region_dispersion': (4, 3), 'region_connectivity': (2, 3)}
    count = counts[family][int(heldout)]
    combination = f'{family}_regions_{count}'
    return tuple(LayoutPrototype(
        f'{group_id}_{role}_proto', family, LAYOUT_FAMILY_CN[family], seed,
        variant, angle, ((0.18 if variant == 0 else 0.42) if family == 'region_coverage' else fraction), count, combination,
    ) for role, variant in (('A', 0), ('B', 1)))


def interior_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """寻找最大内接矩形，让狭长轮廓仍有足够大的可见布局。"""
    inside = np.asarray(mask) >= 250
    heights = [0] * (inside.shape[1] + 1)
    best_area, best = 0, None
    for y, row in enumerate(inside):
        stack = []
        for x in range(len(heights)):
            heights[x] = heights[x] + 1 if x < len(row) and row[x] else 0
            start = x
            while stack and stack[-1][1] > heights[x]:
                left, height = stack.pop()
                width = x - left
                if min(width, height) >= 12 and width * height > best_area:
                    best_area, best = width * height, (left, y - height + 1, width, height)
                start = left
            stack.append((start, heights[x]))
    if best is None:
        raise ValueError('当前轮廓内部没有足以表达布局的可见区域')
    return best


def _field(proto: LayoutPrototype, width: int, height: int) -> np.ndarray:
    """在物体局部坐标构造一种关系，避免使用真实物品模板。"""
    y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    theta = proto.axis_angle
    u, v = x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)
    variant = proto.variant
    count = proto.element_count
    if proto.family == 'region_coverage':
        return -(u ** 2 + v ** 2)
    if proto.family == 'region_position':
        centers = [0.0] if count == 1 else [-0.35, 0.35]
        return -np.minimum.reduce([(u - (-0.38 if variant == 0 else 0.38)) ** 2 + (v - center) ** 2 for center in centers])
    if proto.family == 'region_arrangement':
        major, minor = (u, v) if variant == 0 else (v, u)
        return -np.minimum.reduce([(major - center) ** 2 + minor ** 2 for center in np.linspace(-0.66, 0.66, count)])
    if proto.family == 'region_dispersion':
        spread = 0.20 if variant == 0 else 0.55
        anchors = [(math.cos(t), math.sin(t)) for t in np.linspace(0, math.tau, count, endpoint=False)]
        return -np.minimum.reduce([(u - a * spread) ** 2 + (v - b * spread) ** 2 for a, b in anchors])
    if proto.family == 'region_connectivity':
        if variant == 0:
            return -(v ** 2 + np.maximum(np.abs(u) - 0.55, 0) ** 2)
        return -np.minimum.reduce([(u - center) ** 2 + v ** 2 for center in np.linspace(-0.70, 0.70, count)])
    raise ValueError(f'未知布局关系：{proto.family}')


def layout_map(proto: LayoutPrototype, mask: np.ndarray) -> np.ndarray:
    """在可见内部按固定名次选色，严格匹配同对色块面积。"""
    x, y, width, height = interior_box(mask)
    field = _field(proto, width, height)
    # 面积由共享骨架决定，位置和连接状态不能改变颜色总量。
    count = max(1, round(field.size * proto.region_fraction))
    selected = np.argsort(field.ravel(), kind='stable')[-count:]
    patch = np.zeros(field.size, dtype=np.float32)
    patch[selected] = 1.0
    output = np.full(np.asarray(mask).shape, 0.30, dtype=np.float32)
    output[y:y + height, x:x + width] = 0.30 + 0.65 * patch.reshape(field.shape)
    return output
