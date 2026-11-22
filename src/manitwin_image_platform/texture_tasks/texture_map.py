"""由稳定原型生成纹理场；实例只加入微小成像扰动。"""
import colorsys
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .prototype import TexturePrototype


def _normalize(values: np.ndarray) -> np.ndarray:
    """把有限纹理场映射到单位区间。"""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-7:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _standardize_texture(values: np.ndarray, target_mean: float = 0.5, target_std: float = 0.18) -> np.ndarray:
    """统一纹理全局明暗范围，保留空间结构。"""
    values = np.asarray(values, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return np.full_like(values, target_mean)
    out = (values - mean) / std * target_std + target_mean
    return np.clip(out, 0.0, 1.0)


def _noise(rng: np.random.Generator, width: int, height: int, coarse: int, *, nearest: bool = False) -> np.ndarray:
    """从给定种子生成可重复的多尺度随机场。"""
    cw, ch = max(2, width // coarse), max(2, height // coarse)
    arr = rng.random((ch, cw), dtype=np.float32)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    img = Image.fromarray((arr * 255).astype(np.uint8), mode='L').resize((width, height), resample)
    return np.asarray(img, dtype=np.float32) / 255.0


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    """将共享场景颜色转换为RGB。"""
    return np.array(colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v))), dtype=np.float32)


def _saw(values: np.ndarray) -> np.ndarray:
    """生成连续的三角波。"""
    return 2.0 * np.abs((values / math.tau) % 1.0 - 0.5)


def _crack_field(width: int, height: int, seed: int, thickness: int = 2) -> np.ndarray:
    """固定裂纹的分支骨架。"""
    rng = random.Random(seed)
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(8, 16)):
        x, y = rng.randint(0, width-1), rng.randint(0, height-1)
        pts = [(x, y)]
        for _ in range(rng.randint(3, 7)):
            x = max(0, min(width-1, x + rng.randint(-35, 35)))
            y = max(0, min(height-1, y + rng.randint(-26, 26)))
            pts.append((x, y))
        draw.line(pts, fill=255, width=thickness)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=0.7)), dtype=np.float32) / 255.0


def texture_map(proto: TexturePrototype, width: int, height: int, instance_seed: int) -> np.ndarray:
    """原型决定纹理结构，场景种子只决定微弱的实例噪声。"""
    rng = np.random.default_rng(proto.seed)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    warp_a = _noise(rng, width, height, 10) - 0.5
    warp_b = _noise(rng, width, height, 7) - 0.5
    xw = x + proto.warp * warp_a
    yw = y + proto.warp * warp_b
    f = proto.frequency
    s = proto.secondary_frequency
    phase = proto.phase_bias
    family = proto.family

    if family == 'horizontal_bands':
        values = np.sin(yw * math.pi * f + phase)
    elif family == 'vertical_bands':
        values = np.sin(xw * math.pi * f + phase)
    elif family == 'diagonal_bands_pos':
        values = np.sin((xw + yw) * math.pi * f + phase)
    elif family == 'diagonal_bands_neg':
        values = np.sin((xw - yw) * math.pi * f + phase)
    elif family == 'dots':
        values = np.cos(xw * math.pi * f) * np.cos(yw * math.pi * s)
    elif family == 'speckles':
        values = 1.5 * _noise(rng, width, height, 4) - 0.75
    elif family == 'blobs':
        values = 0.72 * _noise(rng, width, height, 20) + 0.28 * _noise(rng, width, height, 9)
    elif family == 'checker_cells':
        values = np.sign(np.sin(xw * math.pi * f) * np.sin(yw * math.pi * s))
    elif family == 'marble':
        values = np.sin((xw * f + yw * s) * math.pi + 3.0 * _noise(rng, width, height, 13) + phase)
    elif family == 'crosshatch':
        values = 0.5 * (np.sin(xw * math.pi * f + phase) + np.sin(yw * math.pi * s - phase))
    elif family == 'patches':
        values = 0.7 * _noise(rng, width, height, 24) + 0.3 * _noise(rng, width, height, 8)
    elif family == 'broken_waves':
        carrier = np.sin((xw + 0.48*yw) * math.pi * f + phase)
        gate = np.sign(np.cos((xw - yw) * math.pi * s))
        values = carrier * gate
    elif family == 'zigzag':
        coord = (xw + 0.55*yw) * math.pi * f + phase
        values = 1.0 - 2.0 * _saw(coord)
    elif family == 'concentric_rings':
        radius = np.sqrt((xw*1.1)**2 + (yw*0.9)**2)
        values = np.sin(radius * math.pi * f + phase)
    elif family == 'radial_spokes':
        angle = np.arctan2(yw, xw)
        values = np.sin(angle * max(3.0, f) + phase)
    elif family == 'cellular':
        cells = _noise(rng, width, height, 18, nearest=True)
        smooth = _noise(rng, width, height, 18)
        values = 1.4 * np.abs(cells - smooth)
    elif family == 'cracks':
        values = _crack_field(width, height, proto.seed + 17, thickness=max(1, int(1 + proto.thickness*3)))
    elif family == 'woven':
        ax = np.sin(xw * math.pi * f + phase)
        by = np.sin(yw * math.pi * s - phase)
        values = 0.55*ax + 0.45*by + 0.2*ax*by
    elif family == 'dashed_bands':
        carrier = np.sin(yw * math.pi * f + phase)
        dash = (np.sin(xw * math.pi * s * 1.8) > 0).astype(np.float32)
        values = carrier * dash
    elif family == 'chevrons':
        values = np.sin((np.abs(xw) * f + yw * s) * math.pi + phase)
    elif family == 'woodgrain':
        low = _noise(rng, width, height, 16)
        values = np.sin((xw * f + low * 1.8) * math.pi + phase)
    elif family == 'wave_grid':
        a = np.sin((xw + 0.15*np.sin(yw*math.pi*s)) * math.pi * f + phase)
        b = np.sin((yw + 0.15*np.sin(xw*math.pi*f)) * math.pi * s - phase)
        values = 0.5*(a+b)
    elif family == 'stipple_clusters':
        clusters = _noise(rng, width, height, 18)
        specks = _noise(rng, width, height, 3)
        values = clusters * (specks > 0.62).astype(np.float32)
    elif family == 'cellular_rings':
        base = _noise(rng, width, height, 19)
        values = np.sin(base * math.pi * max(3.0, f) + phase)
    else:
        raise ValueError(f'未知纹理家族：{family}')

    values = _normalize(values + (proto.structure - 0.5) * 0.38 * _noise(rng, width, height, 12))
    if family in {'horizontal_bands','vertical_bands','diagonal_bands_pos','diagonal_bands_neg','crosshatch','broken_waves','zigzag','dashed_bands','chevrons'}:
        threshold = 1.0 - proto.thickness
        values = np.where(values > threshold, 1.0, values * (0.28 + 0.48*proto.softness))
    gate = _noise(rng, width, height, 17, nearest=True)
    values = np.where(gate < proto.dropout, values * 0.50, values)
    image = Image.fromarray((np.clip(values,0,1)*255).astype(np.uint8), mode='L')
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.20, 1.45*proto.softness)))
    # Equalize global mean/std so color/mean intensity cannot become a stable class shortcut.
    stable = _standardize_texture(np.asarray(image,dtype=np.float32)/255.0)
    micro = np.random.default_rng(instance_seed).normal(0.0, 0.002, stable.shape)
    return np.clip(stable + micro, 0.0, 1.0).astype(np.float32)


