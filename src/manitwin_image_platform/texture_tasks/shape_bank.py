from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter

TRAIN_SHAPE_FAMILIES = (
    'rounded_rect', 'ellipse', 'capsule', 'trapezoid', 'quadrilateral', 'pentagon', 'hexagon',
    'convex_polygon', 'concave_polygon', 'droplet', 'bean', 'dumbbell', 'l_shape', 'curved_bar',
    'twin_lobe', 'freeform_blob',
)
HELDOUT_SHAPE_FAMILIES = ('crescent', 'cross', 'tri_lobe', 'arrowhead')

SHAPE_FAMILY_CN = {
    'rounded_rect': '圆角长方形', 'ellipse': '椭圆', 'capsule': '胶囊形', 'trapezoid': '梯形',
    'quadrilateral': '随机四边形', 'pentagon': '五边形', 'hexagon': '六边形',
    'convex_polygon': '随机凸多边形', 'concave_polygon': '随机凹多边形', 'droplet': '水滴形',
    'bean': '豆状', 'dumbbell': '哑铃形', 'l_shape': 'L形', 'curved_bar': '弯曲条状',
    'twin_lobe': '双叶形', 'freeform_blob': '自由块状', 'crescent': '月牙形', 'cross': '十字形',
    'tri_lobe': '三叶形', 'arrowhead': '箭头形',
}


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    family: str
    seed: int
    aspect_ratio: float
    roundness: float
    irregularity: float

    @property
    def family_cn(self) -> str:
        """返回界面显示的形状名称。"""
        return SHAPE_FAMILY_CN[self.family]


def make_shape_spec(family: str, seed: int) -> ShapeSpec:
    """用场景种子固定轮廓和长宽比例。"""
    if family not in SHAPE_FAMILY_CN:
        raise ValueError(f'未知形状家族：{family}')
    rng = random.Random(seed)
    return ShapeSpec(
        family=family,
        seed=seed,
        aspect_ratio=rng.uniform(1.0, 2.8),
        roundness=rng.uniform(0.08, 0.48),
        irregularity=rng.uniform(0.05, 0.35),
    )


def _regular_polygon(cx: float, cy: float, rx: float, ry: float, sides: int, phase: float, jitter: float, rng: random.Random) -> list[tuple[float, float]]:
    """构造轻微不规则的多边形轮廓。"""
    points = []
    for i in range(sides):
        angle = phase + math.tau * i / sides + rng.uniform(-0.08, 0.08) * jitter
        scale = 1.0 + rng.uniform(-jitter, jitter)
        points.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    return points


def _smooth_mask(mask: Image.Image, radius: float = 1.15) -> Image.Image:
    """平滑轮廓边缘，减少锯齿。"""
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def render_shape_mask(spec: ShapeSpec, width: int, height: int) -> Image.Image:
    """生成与类别无关的前景轮廓。"""
    if width < 32 or height < 32:
        raise ValueError('形状画布太小')
    rng = random.Random(spec.seed)
    mask = Image.new('L', (width, height), 0)
    d = ImageDraw.Draw(mask)
    pad_x, pad_y = width * 0.08, height * 0.08
    x0, y0, x1, y1 = pad_x, pad_y, width - pad_x, height - pad_y
    cx, cy = width / 2.0, height / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    f = spec.family

    if f == 'rounded_rect':
        d.rounded_rectangle((x0, y0, x1, y1), radius=min(rx, ry) * spec.roundness, fill=255)
    elif f == 'ellipse':
        d.ellipse((x0, y0, x1, y1), fill=255)
    elif f == 'capsule':
        d.rounded_rectangle((x0, y0, x1, y1), radius=min(rx, ry), fill=255)
    elif f == 'trapezoid':
        inset = rx * rng.uniform(0.18, 0.48)
        pts = [(x0 + inset, y0), (x1 - inset * rng.uniform(0.6, 1.2), y0), (x1, y1), (x0, y1)]
        d.polygon(pts, fill=255)
    elif f == 'quadrilateral':
        pts = [(x0 + rng.uniform(0, rx*0.35), y0 + rng.uniform(0, ry*0.2)),
               (x1 - rng.uniform(0, rx*0.28), y0 + rng.uniform(0, ry*0.3)),
               (x1 - rng.uniform(0, rx*0.15), y1 - rng.uniform(0, ry*0.25)),
               (x0 + rng.uniform(0, rx*0.3), y1 - rng.uniform(0, ry*0.18))]
        d.polygon(pts, fill=255)
    elif f in {'pentagon', 'hexagon'}:
        d.polygon(_regular_polygon(cx, cy, rx, ry, 5 if f == 'pentagon' else 6, -math.pi/2, spec.irregularity*0.45, rng), fill=255)
    elif f == 'convex_polygon':
        d.polygon(_regular_polygon(cx, cy, rx, ry, rng.randint(7, 10), rng.uniform(0, math.pi), spec.irregularity*0.7, rng), fill=255)
    elif f == 'concave_polygon':
        sides = rng.randint(8, 12)
        concave_indices = {rng.randrange(sides), rng.randrange(sides)}
        pts = []
        for i in range(sides):
            angle = math.tau * i / sides + rng.uniform(-0.04, 0.04)
            scale = rng.uniform(0.45, 0.7) if i in concave_indices or (i % 4 == 0) else rng.uniform(0.85, 1.03)
            pts.append((cx + math.cos(angle)*rx*scale, cy + math.sin(angle)*ry*scale))
        d.polygon(pts, fill=255)
    elif f == 'droplet':
        pts = [(cx, y0), (x1, cy+ry*0.10), (cx, y1), (x0, cy+ry*0.10)]
        d.polygon(pts, fill=255)
        d.ellipse((x0, cy-ry*0.15, x1, y1), fill=255)
    elif f == 'bean':
        d.ellipse((x0, y0, x1, y1), fill=255)
        cut = Image.new('L', (width, height), 0); cd = ImageDraw.Draw(cut)
        cd.ellipse((cx-rx*0.05, y0+ry*0.05, x1+rx*0.25, cy+ry*0.45), fill=255)
        mask = Image.composite(Image.new('L', (width,height), 0), mask, cut)
    elif f == 'dumbbell':
        r = min(rx, ry) * 0.65
        d.ellipse((x0, cy-r, x0+2*r, cy+r), fill=255)
        d.ellipse((x1-2*r, cy-r, x1, cy+r), fill=255)
        d.rounded_rectangle((x0+r*0.7, cy-r*0.45, x1-r*0.7, cy+r*0.45), radius=r*0.35, fill=255)
    elif f == 'l_shape':
        arm = min(width, height) * rng.uniform(0.22, 0.34)
        d.rounded_rectangle((x0, y0, x0+arm, y1), radius=arm*0.25, fill=255)
        d.rounded_rectangle((x0, y1-arm, x1, y1), radius=arm*0.25, fill=255)
    elif f == 'curved_bar':
        outer = Image.new('L', (width,height), 0); od=ImageDraw.Draw(outer)
        od.arc((x0-rx*0.3, y0-ry*0.2, x1+rx*0.3, y1+ry*0.8), start=205, end=335, fill=255, width=max(6, int(ry*0.9)))
        mask = outer
    elif f == 'twin_lobe':
        r = min(rx, ry)*0.78
        d.ellipse((cx-rx*0.92-r*0.35, cy-r, cx-rx*0.15+r*0.35, cy+r), fill=255)
        d.ellipse((cx+rx*0.15-r*0.35, cy-r, cx+rx*0.92+r*0.35, cy+r), fill=255)
        d.rounded_rectangle((cx-rx*0.45, cy-r*0.55, cx+rx*0.45, cy+r*0.55), radius=r*0.3, fill=255)
    elif f == 'freeform_blob':
        pts = []
        n = rng.randint(10, 16)
        for i in range(n):
            angle = math.tau*i/n
            scale = rng.uniform(0.68, 1.04)
            pts.append((cx+math.cos(angle)*rx*scale, cy+math.sin(angle)*ry*scale))
        d.polygon(pts, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=min(width,height)*0.025))
    elif f == 'crescent':
        d.ellipse((x0, y0, x1, y1), fill=255)
        d.ellipse((cx-rx*0.05, y0+ry*0.02, x1+rx*0.22, y1-ry*0.02), fill=0)
    elif f == 'cross':
        arm = min(width,height)*0.28
        d.rounded_rectangle((cx-arm/2, y0, cx+arm/2, y1), radius=arm*0.2, fill=255)
        d.rounded_rectangle((x0, cy-arm/2, x1, cy+arm/2), radius=arm*0.2, fill=255)
    elif f == 'tri_lobe':
        r = min(rx,ry)*0.62
        centers = [(cx, cy-ry*0.32), (cx-rx*0.38, cy+ry*0.25), (cx+rx*0.38, cy+ry*0.25)]
        for sx,sy in centers: d.ellipse((sx-r,sy-r,sx+r,sy+r), fill=255)
    elif f == 'arrowhead':
        pts = [(x0, cy-ry*0.34), (cx+rx*0.05, cy-ry*0.34), (cx+rx*0.05, y0), (x1, cy), (cx+rx*0.05, y1), (cx+rx*0.05, cy+ry*0.34), (x0, cy+ry*0.34)]
        d.polygon(pts, fill=255)
    else:
        raise ValueError(f'未实现形状：{f}')

    return _smooth_mask(mask, 1.15)
