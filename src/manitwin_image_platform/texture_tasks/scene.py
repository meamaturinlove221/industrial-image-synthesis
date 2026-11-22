from __future__ import annotations

from dataclasses import asdict, dataclass
import random

from .shape_bank import TRAIN_SHAPE_FAMILIES, HELDOUT_SHAPE_FAMILIES, SHAPE_FAMILY_CN

BACKGROUND_STYLES = (
    'soft_gradient', 'radial_gradient', 'cloud_noise', 'coarse_noise',
    'blurred_blocks', 'diagonal_fiber', 'low_contrast_cells', 'industrial_mottle',
)
BACKGROUND_STYLE_CN = {
    'soft_gradient': '柔和渐变', 'radial_gradient': '径向渐变', 'cloud_noise': '低频云雾',
    'coarse_noise': '粗颗粒背景', 'blurred_blocks': '模糊几何块', 'diagonal_fiber': '斜向纤维',
    'low_contrast_cells': '低对比胞状', 'industrial_mottle': '工业斑驳',
}


@dataclass(frozen=True, slots=True)
class SceneSpec:
    scene_id: str
    split: str
    shape_domain: str
    shape_family: str
    shape_family_cn: str
    shape_seed: int
    background_style: str
    background_style_cn: str
    background_seed: int
    base_h: float
    base_s: float
    base_v: float
    accent_h: float
    accent_s: float
    accent_v: float
    angle_deg: float
    center_x: float
    center_y: float
    scale: float
    aspect_scale: float
    object_blur: float
    light_angle_deg: float
    light_strength: float
    shadow_strength: float

    def to_dict(self) -> dict:
        """导出配对共享的场景控制量。"""
        return asdict(self)


def _one_scene(
    scene_id: str,
    split: str,
    shape_domain: str,
    family: str,
    seed: int,
    *,
    background_style: str,
    hue_anchor: float,
    angle_anchor: float,
) -> SceneSpec:
    """创建与类别原型无关的场景。"""
    rng = random.Random(seed)
    if background_style not in BACKGROUND_STYLES:
        raise ValueError(f'未知背景样式：{background_style}')
    # 颜色、角度采用分层抽样：保证同一任务组的 8 个场景真正覆盖不同干扰条件，
    # 而不是依赖纯随机后偶然得到 8 个相似颜色/背景。
    base_h = (hue_anchor + rng.uniform(-0.035, 0.035)) % 1.0
    accent_delta = rng.uniform(0.10, 0.30) * (-1.0 if rng.random() < 0.5 else 1.0)
    # 同一 scene 的两种材质色必须有足够色差，避免“纹理存在但视觉上几乎看不见”。
    # 颜色仍完全由 scene 决定，A/B 共享，因此不会变成类别捷径。
    base_s = rng.uniform(0.16, 0.52)
    base_v = rng.uniform(0.58, 0.88)
    value_sign = -1.0 if base_v > 0.72 else 1.0
    accent_v = max(0.28, min(0.95, base_v + value_sign * rng.uniform(0.20, 0.32)))
    sat_sign = -1.0 if base_s > 0.34 else 1.0
    accent_s = max(0.10, min(0.82, base_s + sat_sign * rng.uniform(0.14, 0.28)))
    bg_style = background_style
    return SceneSpec(
        scene_id=scene_id,
        split=split,
        shape_domain=shape_domain,
        shape_family=family,
        shape_family_cn=SHAPE_FAMILY_CN[family],
        shape_seed=rng.randint(1, 2_000_000_000),
        background_style=bg_style,
        background_style_cn=BACKGROUND_STYLE_CN[bg_style],
        background_seed=rng.randint(1, 2_000_000_000),
        base_h=base_h,
        base_s=base_s,
        base_v=base_v,
        accent_h=(base_h + accent_delta) % 1.0,
        accent_s=accent_s,
        accent_v=accent_v,
        angle_deg=max(-88.0, min(88.0, angle_anchor + rng.uniform(-8.0, 8.0))),
        center_x=rng.uniform(0.43, 0.57),
        center_y=rng.uniform(0.43, 0.57),
        scale=rng.uniform(0.72, 0.98),
        aspect_scale=rng.uniform(0.74, 1.26),
        object_blur=rng.uniform(0.0, 0.95),
        light_angle_deg=rng.uniform(0.0, 360.0),
        light_strength=rng.uniform(0.06, 0.24),
        shadow_strength=rng.uniform(0.10, 0.30),
    )


def build_group_scenes(
    split: str,
    group_index: int,
    images_per_class: int,
    seed: int,
    *,
    heldout_shape_ratio: float = 0.25,
) -> list[SceneSpec]:
    """覆盖多种干扰条件，并打散角度与留出形状的关联。"""
    if images_per_class < 1:
        raise ValueError('每类图片数必须大于 0')
    rng = random.Random(seed + group_index * 1_000_003 + (0 if split == 'train' else 700_001))
    heldout_count = 0 if split == 'train' else int(round(images_per_class * heldout_shape_ratio))
    heldout_count = max(0, min(images_per_class, heldout_count))
    seen_count = images_per_class - heldout_count
    scenes: list[SceneSpec] = []

    # Within a group, use as many distinct shapes as possible. A_i and B_i share this exact SceneSpec.
    train_offset = (group_index * 3) % len(TRAIN_SHAPE_FAMILIES)
    background_offset = (group_index * 5) % len(BACKGROUND_STYLES)
    control_order = rng.sample(range(images_per_class), images_per_class)

    def scene_controls(slot: int) -> tuple[str, float, float]:
        """独立分配场景分层位置，避免留出形状固定对应大角度。"""
        slot = control_order[slot]
        bg = BACKGROUND_STYLES[(background_offset + slot) % len(BACKGROUND_STYLES)]
        hue = ((slot + (group_index % max(1, images_per_class)) / max(1, images_per_class)) / max(1, images_per_class)) % 1.0
        if images_per_class == 1:
            angle = 0.0
        else:
            angle = -78.0 + 156.0 * slot / (images_per_class - 1)
        return bg, hue, angle

    for slot in range(seen_count):
        family = TRAIN_SHAPE_FAMILIES[(train_offset + slot) % len(TRAIN_SHAPE_FAMILIES)]
        scene_seed = rng.randint(1, 2_000_000_000)
        bg, hue, angle = scene_controls(slot)
        scenes.append(_one_scene(
            f'scene_{slot:03d}', split, 'seen', family, scene_seed,
            background_style=bg, hue_anchor=hue, angle_anchor=angle,
        ))
    for j in range(heldout_count):
        slot = seen_count + j
        family = HELDOUT_SHAPE_FAMILIES[(group_index + j) % len(HELDOUT_SHAPE_FAMILIES)]
        scene_seed = rng.randint(1, 2_000_000_000)
        bg, hue, angle = scene_controls(slot)
        scenes.append(_one_scene(
            f'scene_{slot:03d}', split, 'heldout', family, scene_seed,
            background_style=bg, hue_anchor=hue, angle_anchor=angle,
        ))
    return scenes
