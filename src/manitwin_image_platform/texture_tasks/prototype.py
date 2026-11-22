from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

TRAIN_TEXTURE_FAMILIES = (
    'horizontal_bands', 'vertical_bands', 'diagonal_bands_pos', 'diagonal_bands_neg',
    'dots', 'speckles', 'blobs', 'checker_cells', 'marble', 'crosshatch',
    'patches', 'broken_waves', 'zigzag', 'concentric_rings', 'radial_spokes',
    'cellular', 'cracks', 'woven', 'dashed_bands', 'chevrons',
)
HELDOUT_TEXTURE_FAMILIES = ('woodgrain', 'wave_grid', 'stipple_clusters', 'cellular_rings')
ALL_TEXTURE_FAMILIES = TRAIN_TEXTURE_FAMILIES + HELDOUT_TEXTURE_FAMILIES

TEXTURE_FAMILY_CN = {
    'horizontal_bands': '横向带状', 'vertical_bands': '纵向带状',
    'diagonal_bands_pos': '正斜带状', 'diagonal_bands_neg': '反斜带状',
    'dots': '点状', 'speckles': '散点颗粒', 'blobs': '云斑块状',
    'checker_cells': '网格胞状', 'marble': '流纹大理石', 'crosshatch': '交叉纹',
    'patches': '不规则片块', 'broken_waves': '断续波纹', 'zigzag': '锯齿纹',
    'concentric_rings': '同心环纹', 'radial_spokes': '放射纹', 'cellular': '胞状纹',
    'cracks': '裂纹', 'woven': '编织纹', 'dashed_bands': '断续带状', 'chevrons': '人字纹',
    'woodgrain': '木纹', 'wave_grid': '波浪网格', 'stipple_clusters': '簇状点纹',
    'cellular_rings': '环状胞纹',
}


@dataclass(frozen=True, slots=True)
class TexturePrototype:
    prototype_id: str
    family: str
    family_cn: str
    seed: int
    frequency: float
    secondary_frequency: float
    contrast: float
    softness: float
    thickness: float
    warp: float
    dropout: float
    structure: float
    phase_bias: float

    def to_dict(self) -> dict:
        """导出独立于训练代码的纹理原型。"""
        return {**asdict(self), 'prototype_kind': 'texture'}


def sample_texture_prototype(prototype_id: str, family: str, seed: int) -> TexturePrototype:
    """用原型种子固定类别的结构与有效参数。"""
    if family not in ALL_TEXTURE_FAMILIES:
        raise ValueError(f'未知纹理家族：{family}')
    rng = random.Random(seed)
    return TexturePrototype(
        prototype_id=prototype_id,
        family=family,
        family_cn=TEXTURE_FAMILY_CN[family],
        seed=seed,
        frequency=rng.uniform(2.0, 9.5),
        secondary_frequency=rng.uniform(1.3, 7.0),
        contrast=rng.uniform(0.48, 0.78),
        softness=rng.uniform(0.15, 0.78),
        thickness=rng.uniform(0.14, 0.50),
        warp=rng.uniform(0.02, 0.28),
        dropout=rng.uniform(0.01, 0.18),
        structure=rng.uniform(0.16, 0.90),
        phase_bias=rng.uniform(0.0, math.tau),
    )


PARAMETER_RANGES = {
    'frequency': 7.5, 'secondary_frequency': 5.7, 'thickness': 0.36,
    'warp': 0.26, 'structure': 0.74, 'dropout': 0.17, 'softness': 0.63,
}
_NO_COORDINATES = {'speckles', 'blobs', 'patches', 'cellular', 'cracks'}
_TWO_FREQUENCIES = {
    'dots', 'checker_cells', 'marble', 'crosshatch', 'broken_waves',
    'woven', 'dashed_bands', 'chevrons', 'wave_grid',
}
_THICKNESS_FAMILIES = {
    'horizontal_bands', 'vertical_bands', 'diagonal_bands_pos',
    'diagonal_bands_neg', 'crosshatch', 'broken_waves', 'zigzag',
    'dashed_bands', 'chevrons', 'cracks',
}


def effective_parameters(family: str) -> tuple[str, ...]:
    """只返回本纹理公式实际使用的参数，排除虚假的距离贡献。"""
    names = ['structure', 'dropout', 'softness']
    if family not in _NO_COORDINATES and family != 'stipple_clusters':
        names += ['frequency']
        if family != 'cellular_rings':
            names += ['warp']
    if family in _TWO_FREQUENCIES:
        names.append('secondary_frequency')
    if family in _THICKNESS_FAMILIES:
        names.append('thickness')
    return tuple(names)


def prototype_distance(a: TexturePrototype, b: TexturePrototype) -> float:
    """按家族有效参数计算距离，布局原型由其状态定义差异。"""
    if a.family != b.family:
        return 1.0
    if hasattr(a, 'variant'):
        return float(a.variant != b.variant)
    diffs = [abs(getattr(a, name) - getattr(b, name)) / PARAMETER_RANGES[name]
             for name in effective_parameters(a.family)]
    return float(sum(diffs) / len(diffs))


def make_distinct_same_family_pair(group_id: str, family: str, seed_a: int, seed_b: int, min_distance: float = 0.22) -> tuple[TexturePrototype, TexturePrototype]:
    """构造有效参数足够分离的同家族纹理对。"""
    a = sample_texture_prototype(f'{group_id}_A_proto', family, seed_a)
    attempt_seed = seed_b
    for _ in range(64):
        b = sample_texture_prototype(f'{group_id}_B_proto', family, attempt_seed)
        if prototype_distance(a, b) >= min_distance:
            return a, b
        attempt_seed += 104729
    raise RuntimeError(f'无法为 {group_id} 构造足够不同的同家族纹理对')
